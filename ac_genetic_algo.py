import shutil
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from os import getpid
from pathlib import Path
import csv
import onnx
import random
import torch
from onnx import shape_inference
from onnxruntime.training import artifacts
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, Problem, StarmapParallelization
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from model.resnet18 import ResNet18
from process_onnx import (
    split_forward_backward,
)
from test_ac import apply_onnx_pass, remove_checkpoint
from tools import run_stream


# TODO: check if the forward outputs and inputs do not need to be recomputed at each pass
def apply_activation_checkpointing(model, recomputations, forward_outputs, forward_inputs):
    for recomputation in recomputations:
        model, compute_cost = remove_checkpoint(model, recomputation, forward_outputs, forward_inputs)
        forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(model)
    return model


class ActivationCheckpointingProblem(Problem):
    def __init__(
        self,
        optimization_vars,
        forward_inputs,
        forward_outputs,
        model_path,
        accelerator_path,
        mapping_path,
        output_path,
        processes,
    ):
        self.optimization_vars = optimization_vars
        self.model_path = model_path
        self.accelerator_path = accelerator_path
        self.mapping_path = mapping_path
        self.output_path = output_path
        self.forward_inputs = forward_inputs
        self.forward_outputs = forward_outputs

        # Parallelization
        self.processes = processes

        super().__init__(
            n_var=len(optimization_vars),  # Length of binary string
            n_obj=3,  # Number of objectives
            n_constr=0,  # No constraints
            xl=0,  # Lower bound (binary)
            xu=1,  # Upper bound (binary)
            elementwise_evaluation=False,
        )

    def single_stream_eval(self, x):
        # Get the process ID for tracking
        pid = getpid()
        folder = f"{output_path}{pid}/"
        Path(folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_path, f"{folder}model.onnx")
        # Generate the ONNX based on X
        recomputations = []
        for variable, activations in zip(x, optimization_vars, strict=True):
            if variable:
                recomputations.append([activations, optimization_vars[activations]])

        onnx_model = onnx.load(f"{folder}model.onnx")
        checkpointed_model = apply_activation_checkpointing(
            onnx_model, recomputations, self.forward_outputs, self.forward_inputs
        )
        onnx.save(checkpointed_model, f"{folder}checkpointed.onnx")
        processed_model_path, forward_path, backward_pass, opt_pass = apply_onnx_pass(
            output_path=f"{folder}/", model=checkpointed_model
        )

        # Evaluate with Stream
        latency, energy, memory = run_stream(
            processed_model_path, self.accelerator_path, self.mapping_path, x, f"{folder}"
        )
        return latency, energy, memory

    def _evaluate(self, x, out, *args, **kwargs):
        with Pool(processes=self.processes) as pool:
            r = pool.map(self.single_stream_eval, [individual for individual in x])

        out["F"] = r
        # the objectives are the energy, latency and memory


def generate_model(output_path):
    model_path = f"{output_path}model.onnx"
    train_onnx_path = f"{output_path}training_model.onnx"
    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18()
    for param in model.parameters():
        if param.dim() > 1:  # Weights
            torch.nn.init.kaiming_uniform_(param)
        else:
            torch.nn.init.uniform(param, 3, 4)
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, model_path, opset_version=13)
    shape_inference.infer_shapes_path(model_path, model_path)

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
    inferred_model = shape_inference.infer_shapes(
        shape_inference.infer_shapes(shape_inference.infer_shapes(onnx.load(train_onnx_path)))
    )
    onnx.save(inferred_model, train_onnx_path)

    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(inferred_model)
    # The input and the loss is not an activation to be checkpointed
    optimization_vars = {}
    for key, item in forward_outputs.items():
        if "lazy_reset" not in key and "loss" not in key and "prob" not in key:
            optimization_vars[key] = item
    return optimization_vars, train_onnx_path, forward_inputs, forward_outputs


def test(output_path):
    optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(output_path)

    n = len(optimization_vars)
    x = [False, False, False, True, True]
    # Generate the ONNX based on X
    recomputations = []
    for variable, activations in zip(x, optimization_vars, strict=True):
        if variable:
            recomputations.append([activations, optimization_vars[activations]])

    onnx_model = onnx.load(model_path)
    checkpointed_model = apply_activation_checkpointing(onnx_model, recomputations, forward_outputs, forward_inputs)
    onnx.save(checkpointed_model, f"{output_path}checkpointed.onnx")

    return 0


if __name__ == "__main__":
    accelerator_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
    output_path = "results/ga_ac/"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(output_path)

    # Run the optimization
    problem = ActivationCheckpointingProblem(
        optimization_vars,
        forward_inputs,
        forward_outputs,
        model_path,
        accelerator_path,
        mapping_path,
        output_path,
        processes=32,
    )

    algorithm = NSGA2(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=BinomialCrossover(n_offsprings=2, prob=0.9),
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 50),  # Number of generations
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
        writer.writerow(best_x + best_f)
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
    # Plot the Pareto front
    Scatter().add(res.F).show()
