import argparse
import csv
import logging
import os
from pathlib import Path

import torch
from onnxruntime.training import artifacts
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

import onnx
from ac_genetic_algo import ActivationCheckpointingProblem
from model.resnet224 import ResNet18_224
from process_onnx import split_forward_backward
from tools import apply_onnx_passes, run_stream


def argparser():
    parser = argparse.ArgumentParser(description="Stream Hardware Search for ResNet18")
    parser.add_argument("--output_path", type=str, default="onnx/output/", help="Path to the output directory")
    parser.add_argument("--processes", type=int, required=True, help="number of processes")

    parser.add_argument(
        "--accelerator_path",
        type=str,
        default="stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        help="Path to Stream Accelerator",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise2.yaml",
        help="Path to Stream Mapping file",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch SIze to evaluate the neural networks")

    # GA hyperparameters
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--n_offsprings", type=int, default=2)
    parser.add_argument("--offspring_prob", type=int, default=0.9)
    parser.add_argument("--bitflip_prob", type=int, default=0.1)
    return parser.parse_args()


def generate_model(output_path, batch_size=1):
    model_path: str = f"{output_path}model.onnx"
    train_onnx_path = f"{output_path}training_model.onnx"
    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18_224()
    for param in model.parameters():
        if param.dim() > 1:  # Weights
            torch.nn.init.kaiming_uniform_(param)
        else:
            torch.nn.init.uniform(param, 3, 4)
    torch_input = torch.randn(batch_size, 3, 224, 224)
    torch.onnx.export(model, torch_input, model_path, opset_version=13)
    onnx.shape_inference.infer_shapes_path(model_path, model_path)

    onnx_model = onnx.load(model_path)
    inits = onnx_model.graph.initializer
    requires_grad = []
    for init in inits:
        requires_grad.append(init.name)

    loss = artifacts.LossType(2)
    artifacts.generate_artifacts(
        onnx_model, requires_grad=requires_grad, loss=loss, optimizer=artifacts.OptimType.AdamW, prefix=output_path
    )

    # Multiple shapes inference pass are needed
    inferred_model = onnx.shape_inference.infer_shapes(
        onnx.shape_inference.infer_shapes(onnx.shape_inference.infer_shapes(onnx.load(train_onnx_path)))
    )
    onnx.save(inferred_model, train_onnx_path)

    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(inferred_model)
    # The input and the loss is not an activation to be checkpointed
    optimization_vars = {}
    for key, item in forward_outputs.items():
        if "lazy_reset" not in key and "loss" not in key and "prob" not in key:
            optimization_vars[key] = item
    return optimization_vars, train_onnx_path, forward_inputs, forward_outputs


def evaluate_activation_checkpointing(
    optimization_vars,
    forward_inputs,
    forward_outputs,
    output_path,
    model_path,
    args,
    mode="fused",
):
    problem = ActivationCheckpointingProblem(
        optimization_vars,
        forward_inputs,
        forward_outputs,
        model_path,
        args.accelerator_path,
        args.mapping_path,
        output_path,
        processes=args.processes,
        mode=mode,
    )

    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=BinaryRandomSampling(),
        crossover=BinomialCrossover(n_offsprings=args.n_offsprings, prob=args.offspring_prob),
        mutation=BitflipMutation(prob=args.bitflip_prob),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", args.generations),  # Number of generations
        seed=1,
        verbose=True,
        save_history=True,
    )

    best_x = res.X
    best_f = res.F
    best_pop_x = res.pop.get("X")
    best_pop_f = res.pop.get("F")

    with open(f"{output_path}result.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(best_x)
        writer.writerow(best_f)
        for i, (ind_x, ind_f) in enumerate(zip(best_pop_x, best_pop_f, strict=True)):
            writer.writerow([i] + ind_x.tolist() + ind_f.tolist())
    history_list = []
    for i, run in enumerate(res.history):
        pop = run.pop
        for j, individual in enumerate(pop):
            temp_list = [i, j] + individual._X.tolist() + individual._F.tolist()
            history_list.append(temp_list)

    with open(f"{output_path}history.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Individual", "X", "F"])
        for line in history_list:
            writer.writerow(line)

    logging.critical(f"{best_pop_f}, {best_pop_x}, {best_x}, {best_f}")


def main(args):
    # base config
    # output_path = os.path.join(args.output_path, "Base/")
    # Path(output_path).mkdir(parents=True, exist_ok=True)
    # onnx_path = f"{output_path}/test.onnx"
    # infered_path = f"{output_path}/inferred.onnx"
    # model = ResNet18_224()
    # torch_input = torch.randn(args.batch_size, 3, 224, 224)
    # torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    # onnx.shape_inference.infer_shapes_path(onnx_path, infered_path)

    # # Generate Backward
    # base_model = onnx.load(infered_path)
    # inits = base_model.graph.initializer
    # requires_grad = []
    # for init in inits:
    #     requires_grad.append(init.name)

    # # layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    # inferred_train_onnx_path4, forward_path, _, _ = apply_onnx_passes(
    #     base_model, None, output_path, requires_grad, "onnx", check=False
    # )
    # layer_stacks = None
    # energy, latency, memory = run_stream(
    #     inferred_train_onnx_path4,
    #     args.accelerator_path,
    #     args.mapping_path,
    #     id=1,
    #     output_path=output_path,
    #     mode="lbl",
    #     layer_stacks=layer_stacks,
    # )
    # logging.critical(f"{energy}, {latency}, {memory}")
    # base config + fused
    output_path = os.path.join(args.output_path, "Base_Fused/")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    onnx_path = f"{output_path}/test.onnx"
    infered_path = f"{output_path}/inferred.onnx"
    model = ResNet18_224()
    torch_input = torch.randn(args.batch_size, 3, 224, 224)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    onnx.shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        requires_grad.append(init.name)

    # layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    inferred_train_onnx_path4, forward_path, _, _ = apply_onnx_passes(
        base_model, None, output_path, requires_grad, "onnx", check=False
    )
    layer_stacks = None
    energy, latency, memory = run_stream(
        inferred_train_onnx_path4,
        args.accelerator_path,
        args.mapping_path,
        id=1,
        output_path=output_path,
        mode="fused",
        layer_stacks=layer_stacks,
    )
    logging.critical(f"{energy}, {latency}, {memory}")
    # # base config + AC
    # output_path = os.path.join(args.output_path, "Base_AC/")
    # Path(output_path).mkdir(parents=True, exist_ok=True)
    # optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(
    #     output_path, batch_size=args.batch_size
    # )
    # evaluate_activation_checkpointing(
    #     optimization_vars=optimization_vars,
    #     forward_inputs=forward_inputs,
    #     forward_outputs=forward_outputs,
    #     output_path=output_path,
    #     model_path=model_path,
    #     args=args,
    #     mode="lbl",
    # )
    # base config + AC + fused
    # output_path = os.path.join(args.output_path, "Base_AC_Fused/")
    # Path(output_path).mkdir(parents=True, exist_ok=True)
    # optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(
    #     output_path, batch_size=args.batch_size
    # )
    # evaluate_activation_checkpointing(
    #     optimization_vars=optimization_vars,
    #     forward_inputs=forward_inputs,
    #     forward_outputs=forward_outputs,
    #     output_path=output_path,
    #     model_path=model_path,
    #     args=args,
    #     mode="fused",
    # )


if __name__ == "__main__":
    args = argparser()
    main(args)
