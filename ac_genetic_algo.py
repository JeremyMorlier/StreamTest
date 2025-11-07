import csv
import logging
import os
import shutil
import traceback
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os import getpid
from pathlib import Path
from typing import Literal

import onnx
import onnxruntime as ort
import torch
from onnx import shape_inference
from onnxruntime.training import artifacts
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from stream.api import _sanity_check_inputs
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_save

from model.resnet18 import ResNet18
from model.resnet224 import ResNet18_224
from process_onnx import (
    split_forward_backward,
)
from test_ac import apply_onnx_pass, remove_checkpoint
from tools import get_max_offchip_memory, run_stream


def apply_activation_checkpointing(model, recomputations, forward_outputs, forward_inputs):
    local_forward_inputs, local_forward_outputs = (
        forward_inputs,
        forward_outputs,
    )
    total_memory_cost = 0
    for recomputation in recomputations:
        model, compute_cost, memory_cost = remove_checkpoint(
            model, recomputation, local_forward_outputs, local_forward_inputs
        )
        local_forward_inputs, _, local_forward_outputs, _ = split_forward_backward(model)
        total_memory_cost += memory_cost
    return model, total_memory_cost


def bool_list_to_string(bool_list):
    return "".join(["1" if b else "0" for b in bool_list])


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

        latency, energy, memory, saved_memory = 0, 0, 0, 0
        try:
            # Generate the ONNX based on X
            recomputations = []
            for variable, activations in zip(x, optimization_vars, strict=True):
                if variable:
                    recomputations.append([activations, optimization_vars[activations]])

            recomputations.reverse()
            onnx_model = onnx.load(f"{folder}model.onnx")
            checkpointed_model, saved_memory = apply_activation_checkpointing(
                onnx_model, recomputations, self.forward_outputs, self.forward_inputs
            )
            onnx.save(checkpointed_model, f"{folder}checkpointed.onnx")
            processed_model_path, forward_path, backward_pass, opt_pass = apply_onnx_pass(
                output_path=f"{folder}/", model=checkpointed_model
            )

            # Evaluate with Stream
            latency, energy, memory = optimize_allocation_ga_no_id(
                self.accelerator_path,
                processed_model_path,
                self.mapping_path,
                mode="fused",
                layer_stacks=None,
                nb_ga_generations=4,
                nb_ga_individuals=4,
                output_path=f"{folder}",
                id=bool_list_to_string(x),
            )
        except Exception as e:
            error_msg = traceback.format_exc()
            logging.error(f"{e}, trace {error_msg}")
        logging.error(f"{id} {latency} {energy}, {saved_memory}")
        return latency, energy, -saved_memory

    def _evaluate(self, x, out, *args, **kwargs):
        with Pool(processes=self.processes) as pool:
            r = pool.map(self.single_stream_eval, [individual for individual in x])

        out["F"] = r
        # the objectives are the energy, latency and memory


def optimize_allocation_ga_no_id(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    nb_ga_generations: int,
    nb_ga_individuals: int,
    output_path: str,
    id: str,
    temporal_mapping_type: str = "uneven",
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    # Create experiment_id path
    os.makedirs(f"{output_path}{id}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}{id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}{id}/cost_lut.pickle"
    scme_path = f"{output_path}{id}/scme.pickle"
    allocations_path = f"{output_path}/waco/"
    cost_lut_post_co_path = f"{output_path}/cost_lut_post_co.pickle"
    tiled_workload_post_co_path = f"{output_path}/tiled_workload_post_co.pickle"

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    mainstage = MainStage(
        [  # Initializes the MainStage as entry point
            AcceleratorParserStage,  # Parses the accelerator
            StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
            LayerStacksGenerationStage,
            TilingGenerationStage,
            TiledWorkloadGenerationStage,
            ZigZagCoreMappingEstimationStage,
            SetFixedAllocationPerformanceStage,
            SchedulingOrderGenerationStage,
            GeneticAlgorithmAllocationStage,
        ],
        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaEngine
        nb_ga_generations=nb_ga_generations,  # number of genetic algorithm (ga) generations
        nb_ga_individuals=nb_ga_individuals,  # number of individuals in each ga generation
        mode=mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=tiled_workload_path,
        cost_lut_path=cost_lut_path,
        allocations_path=allocations_path,
        tiled_workload_post_co_path=tiled_workload_post_co_path,
        cost_lut_post_co_path=cost_lut_post_co_path,
        temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
        operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
    )
    # Launch the MainStage
    answers = mainstage.run()
    scme = answers[0][0]
    pickle_save(scme, scme_path)  # type: ignore
    memory = get_max_offchip_memory(scme)
    return scme.latency, scme.energy, memory


def generate_model(output_path):
    model_path: str = f"{output_path}model.onnx"
    train_onnx_path = f"{output_path}training_model.onnx"
    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18_224()
    for param in model.parameters():
        if param.dim() > 1:  # Weights
            torch.nn.init.kaiming_uniform_(param)
        else:
            torch.nn.init.uniform(param, 3, 4)
    torch_input = torch.randn(1, 3, 224, 224)
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
    import random

    n = len(optimization_vars)
    x = random.choices([False, True], k=n)
    # x =[False, True, True, False, False]
    # Generate the ONNX based on X
    recomputations = []
    for variable, activations in zip(x, optimization_vars, strict=True):
        if variable:
            recomputations.append([activations, optimization_vars[activations]])
    # print([(x, [node.name for node in node_li]) for x, node_li in recomputations])
    recomputations.reverse()
    # print([(x, [node.name for node in node_li]) for x, node_li in recomputations])
    onnx_model = onnx.load(model_path)
    checkpointed_model = apply_activation_checkpointing(onnx_model, recomputations, forward_outputs, forward_inputs)
    onnx.save(checkpointed_model, f"{output_path}checkpointed.onnx")
    processed_model_path, forward_path, backward_pass, opt_pass = apply_onnx_pass(
        output_path=f"{output_path}/", model=checkpointed_model
    )
    accelerator_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise2.yaml"
    # Evaluate with Stream
    latency, energy, memory = optimize_allocation_ga_no_id(
        accelerator_path,
        processed_model_path,
        mapping_path,
        mode="fused",
        layer_stacks=None,
        nb_ga_generations=4,
        nb_ga_individuals=4,
        output_path=f"{output_path}",
        id=x,
    )
    return latency, energy, memory

    return 0


if __name__ == "__main__":
    accelerator_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise.yaml"
    output_path = "results/ga_ac/"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(output_path)

    ort.set_default_logger_severity(3)

    _logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=_logging_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.CRITICAL)
    error_handler = logging.FileHandler("error.log")
    error_handler.setLevel(logging.ERROR)
    info_handler = logging.FileHandler("log.log")
    info_handler.setLevel(logging.INFO)
    warning_handler = logging.FileHandler("warning.log")
    warning_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().addHandler(error_handler)
    logging.getLogger().addHandler(info_handler)
    logging.getLogger().addHandler(warning_handler)

    # test(output_path)
    # Run the optimization
    problem = ActivationCheckpointingProblem(
        optimization_vars,
        forward_inputs,
        forward_outputs,
        model_path,
        accelerator_path,
        mapping_path,
        output_path,
        processes=2,
    )

    algorithm = NSGA2(
        pop_size=20,
        sampling=BinaryRandomSampling(),
        crossover=BinomialCrossover(n_offsprings=2, prob=0.9),
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 6),  # Number of generations
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
    try:
        # Plot the Pareto front
        Scatter().add(res.F).show()
    except Exception as e:
        logging.error(e)

    logging.critical(best_pop_f, best_pop_x, best_x, best_f)
