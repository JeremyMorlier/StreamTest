import csv
import json
import logging
import os
import shutil
from itertools import combinations
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os import getpid
from pathlib import Path
from typing import Literal

import onnxruntime as ort
import torch
from onnxruntime.training import artifacts
from stream.api import _sanity_check_inputs
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
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
from zigzag.utils import pickle_load, pickle_save
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json


import onnx
from model.resnet18 import ResNet18
from model.resnet224 import ResNet18_224
from onnx import shape_inference
from process_onnx import (
    split_forward_backward,
)
from test_ac import apply_onnx_pass, remove_checkpoint
from tools import get_max_offchip_memory, run_stream

ort.set_default_logger_severity(3)


# TODO: check if the forward outputs and inputs do not need to be recomputed at each pass
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


def save_dict_to_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)  # `indent=4` for pretty formatting


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
    logging.error(f"{id} {scme.latency} {scme.energy}, {memory}")

    cost_lut = CostModelEvaluationLUT(cost_lut_path)
    print(cost_lut)
    print(cost_lut.__dict__)
    print(cost_lut_path, os.path.exists(cost_lut_path))
    print(scme.latency, type(scme.latency))
    with open(f"{output_path}/resultt.txt", "a") as f:
        f.write(f"{scme.energy}    {scme.latency} \n")
    # Plotting schedule timeline of best SCME
    # scme.plot_schedule(
    #     plot_full_schedule=True,
    #     draw_dependencies=True,
    #     plot_data_transfer=True,
    #     fig_path=f"{output_path}/{id}/schedule.html",
    # )

    try:
        # Plotting memory usage of best SCME
        scme.plot_memory_usage((0,), (100,), fig_path=f"{output_path}/{id}/memory.png")
    except Exception as e:
        print(e)

    try:
        # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
        convert_scme_to_perfetto_json(scme, cost_lut, json_path=f"{output_path}/{id}/scme.json")
    except Exception as e:
        print(e)
    memory = get_max_offchip_memory(scme)
    return scme.latency, scme.energy, memory


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
    torch_input = torch.randn(32, 3, 32, 32)
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


def diagonal_true_matrix(n):
    return [[(i == j) for j in range(n)] for i in range(n)]


def generate_boolean_variants(N):
    # Generate all combinations of two indices where True will be placed
    for indices in combinations(range(N), 2):
        # Create a list of False values
        variant = [False] * N
        # Set the selected indices to True
        for index in indices:
            variant[index] = True
        yield variant


def single_stream_eval(
    x, model_path, output_path, optimization_vars, forward_outputs, forward_inputs, accelerator_path, mapping_path
):
    # Get the process ID for tracking
    pid = getpid()
    folder = f"{output_path}{pid}/"
    Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, f"{folder}model.onnx")

    latency, energy, memory = 0, 0, 0
    try:
        # Generate the ONNX based on X
        recomputations = []
        for variable, activations in zip(x, optimization_vars, strict=True):
            if variable:
                recomputations.append([activations, optimization_vars[activations]])

        recomputations.reverse()
        onnx_model = onnx.load(f"{folder}model.onnx")
        checkpointed_model, total_memory_cost = apply_activation_checkpointing(
            onnx_model, recomputations, forward_outputs, forward_inputs
        )
        onnx.save(checkpointed_model, f"{folder}checkpointed.onnx")
        processed_model_path, forward_path, backward_pass, opt_pass = apply_onnx_pass(
            output_path=f"{folder}/", model=checkpointed_model
        )

        # Evaluate with Stream
        latency, energy, memory = optimize_allocation_ga_no_id(
            accelerator_path,
            processed_model_path,
            mapping_path,
            mode="fused",
            layer_stacks=None,
            nb_ga_generations=4,
            nb_ga_individuals=4,
            output_path=f"{folder}",
            id=bool_list_to_string(x),
        )
    except Exception as e:
        logging.error(e)
    return x, latency, energy, memory, total_memory_cost


def multiprocess_evaluations(
    x,
    out,
    processes,
    model_path,
    output_path,
    optimization_vars,
    forward_outputs,
    forward_inputs,
    accelerator_path,
    mapping_path,
):
    # Prepare the arguments for each individual in x
    args = [
        (
            individual,
            model_path,
            output_path,
            optimization_vars,
            forward_outputs,
            forward_inputs,
            accelerator_path,
            mapping_path,
        )
        for individual in x
    ]

    # Use Pool to parallelize the evaluations
    with Pool(processes=processes) as pool:
        r = pool.starmap(single_stream_eval, args)
    # the objectives are the energy, latency and memory
    # Assign the results to the output dictionary
    out["F"] = r


if __name__ == "__main__":
    accelerator_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise2.yaml"
    output_path = "results/verify_ac_3/"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(output_path)

    # logging.disable(logging.INFO)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.CRITICAL)
    # error_handler = logging.FileHandler("error2.log")
    # error_handler.setLevel(logging.ERROR)
    # info_handler = logging.FileHandler("log2.log")
    # info_handler.setLevel(logging.INFO)
    # logging.getLogger().addHandler(stream_handler)
    # logging.getLogger().addHandler(error_handler)
    # logging.getLogger().addHandler(info_handler)
    # test(output_path)

    n_vars = len(optimization_vars)
    out_x = {}
    x = diagonal_true_matrix(n_vars)
    print(x)
    # x = [
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    #     True,
    # ]
    # print(
    #     single_stream_eval(
    #         x,
    #         model_path,
    #         output_path,
    #         optimization_vars,
    #         forward_outputs,
    #         forward_inputs,
    #         accelerator_path,
    #         mapping_path,
    #     )
    # )
    x = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    print(
        single_stream_eval(
            x,
            model_path,
            output_path,
            optimization_vars,
            forward_outputs,
            forward_inputs,
            accelerator_path,
            mapping_path,
        )
    )
    x = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    print(
        single_stream_eval(
            x,
            model_path,
            output_path,
            optimization_vars,
            forward_outputs,
            forward_inputs,
            accelerator_path,
            mapping_path,
        )
    )
    x = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    print(
        single_stream_eval(
            x,
            model_path,
            output_path,
            optimization_vars,
            forward_outputs,
            forward_inputs,
            accelerator_path,
            mapping_path,
        )
    )
    x = [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    print(
        single_stream_eval(
            x,
            model_path,
            output_path,
            optimization_vars,
            forward_outputs,
            forward_inputs,
            accelerator_path,
            mapping_path,
        )
    )
    # save_dict_to_json(out_x, "single_val.txt")
    # multiprocess_evaluations(
    #     x,
    #     out_x,
    #     2,
    #     model_path,
    #     output_path,
    #     optimization_vars,
    #     forward_outputs,
    #     forward_inputs,
    #     accelerator_path,
    #     mapping_path,
    # )

    # out_x2 = {}
    # iter_boolean = generate_boolean_variants(n_vars)
    # x2 = [element for element in iter_boolean]
    # multiprocess_evaluations(
    #     x,
    #     out_x2,
    #     2,
    #     model_path,
    #     output_path,
    #     optimization_vars,
    #     forward_outputs,
    #     forward_inputs,
    #     accelerator_path,
    #     mapping_path,
    # )
    # save_dict_to_json(out_x2, "single_duo.txt")
    # print(out_x2)
