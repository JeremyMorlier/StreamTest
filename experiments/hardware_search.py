import argparse
import json
import logging as _logging
import math
import multiprocessing
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import onnx
import onnxruntime as ort
import torch
from onnx import shape_inference
from onnxruntime.training import artifacts
from onnxsim import simplify
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
from zigzag.utils import pickle_load, pickle_save

from model.resnet18 import ResNet18
from streamtest.hardware import (
    stream_edge_tpu,
    stream_edge_tpu_core,
    stream_edge_tpu_mapping,
    to_yaml,
)
from streamtest.onnx_processing import (
    process_1d_nodes,
    process_convolution_grad,
    process_poolgrad,
    split_forward_backward,
)

_logging.basicConfig(level=_logging.ERROR)
# Set the logging level to ERROR to suppress warnings
ort.set_default_logger_severity(3)


def argparser():
    parser = argparse.ArgumentParser(description="Stream Hardware Search for ResNet18")
    parser.add_argument("--output_path", type=str, default="onnx/output/", help="Path to the output directory")
    parser.add_argument("--mode", type=str, default="fused", help="Stream mode (fused or lbl)")

    return parser.parse_args()


def sample_hardware_configs(choices):
    hardware_config = {
        "n_SIMDS": random.choice(choices["n_SIMDS"]),
        "n_computes_lanes": random.choice(choices["n_computes_lanes"]),
        "PE_Memory": random.choice(choices["PE_Memory"]),
        "register_file_size": random.choice(choices["register_file_size"]),
        "xPE": random.choice(choices["xPE"]),
        "yPE": random.choice(choices["yPE"]),
    }
    return hardware_config


def optimize_allocation_ga_no_id(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    nb_ga_generations: int,
    nb_ga_individuals: int,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    # Create experiment_id path
    os.makedirs(f"{output_path}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/cost_lut.pickle"
    scme_path = f"{output_path}/scme.pickle"

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
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
            temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
            operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)  # type: ignore
    return scme


class ConfigGenerator:
    def __init__(self, max_iter, hw_choices, mapping_config, hardware_config, path, nn_path, mode):
        self.max_iter = max_iter
        self.i = 0
        self.hw_choices = hw_choices

        self.mapping_config = mapping_config
        self.hardware_config = hardware_config
        self.path = path
        self.nn_path = nn_path
        self.mode = mode

    def __next__(self):
        config = {}
        if self.i < self.max_iter:
            self.i += 1

            config["hardware_config"] = sample_hardware_configs(self.hw_choices)
            config["mapping_config"] = self.mapping_config
            config["path"] = self.path
            config["mode"] = self.mode
            config["forward_backward"] = f"{self.nn_path}/forward_backward.onnx"
            config["forward"] = f"{self.nn_path}/forward.onnx"
            config["backward"] = f"{self.nn_path}/backward.onnx"
            config["id"] = str(self.i)

            return config
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_iter


def evaluate_performance(config):
    result = {}
    hardware_config = config["hardware_config"]
    mode = config["mode"]
    id_process = multiprocessing.current_process().name
    id_process = id_process.split("-")[-1]
    folder = config["path"] + "/" + id_process + "/" + config["id"]
    Path(folder).mkdir(parents=True, exist_ok=True)

    forward_backward_path = config["forward_backward"]
    forward_path = config["forward"]
    backward_path = config["backward"]

    # Generate Hardware and Mapping Config
    core = stream_edge_tpu_core(
        hardware_config["n_SIMDS"],
        hardware_config["n_computes_lanes"],
        hardware_config["PE_Memory"],
        hardware_config["register_file_size"],
    )
    soc = stream_edge_tpu(
        hardware_config["xPE"],
        hardware_config["yPE"],
        core,
        ["pooling.yaml", "simd.yaml"],
        "offchip.yaml",
        32,
        0.0,
    )

    mapping = stream_edge_tpu_mapping(hardware_config["xPE"], hardware_config["yPE"], ["pooling.yaml", "simd.yaml"])

    # Save Configs in a file to preserve the results
    to_yaml(core, f"{folder}/core.yaml")
    to_yaml(soc, f"{folder}/hardware_config.yaml")
    to_yaml(mapping, f"{folder}/mapping_config.yaml")

    # Evaluate Using Stream
    try:
        scme = optimize_allocation_ga_no_id(
            hardware=f"{folder}/hardware_config.yaml",
            workload=forward_backward_path,
            mapping=f"{folder}/mapping_config.yaml",
            mode=mode,
            layer_stacks=None,
            nb_ga_generations=2,
            nb_ga_individuals=2,
            output_path=f"{folder}/{mode}/",
            skip_if_exists=False,
        )
        result["forward_backward"] = [scme.energy, scme.latency]
        # Memory is not directly available for fused mode

        scme = optimize_allocation_ga_no_id(
            hardware=f"{folder}/hardware_config.yaml",
            workload=forward_path,
            mapping=f"{folder}/mapping_config.yaml",
            mode=mode,
            layer_stacks=None,
            nb_ga_generations=2,
            nb_ga_individuals=2,
            output_path=f"{folder}/forward/",
            skip_if_exists=False,
        )
        result["forward"] = [scme.energy, scme.latency]

        scme = optimize_allocation_ga_no_id(
            hardware=f"{folder}/hardware_config.yaml",
            workload=backward_path,
            mapping=f"{folder}/mapping_config.yaml",
            mode=mode,
            layer_stacks=None,
            nb_ga_generations=2,
            nb_ga_individuals=2,
            output_path=f"{folder}/backward/",
            skip_if_exists=False,
        )
        result["backward"] = [scme.energy, scme.latency]

        result["id"] = config["id"]
        result["hardware_config"] = hardware_config
        result["mapping_config"] = config["mapping_config"]
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {}

    return result


def generate_resnet18_onnx(output_path):
    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, f"{output_path}/resnet18.onnx", opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(
        f"{output_path}/resnet18.onnx", f"{output_path}/resnet18.onnx"
    )

    # Generate Backward
    base_model = onnx.load(f"{output_path}/resnet18.onnx")
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        # if len(init.dims) != 1 :
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)

    artifacts.generate_artifacts(
        base_model, requires_grad=requires_grad, loss=loss, optimizer=artifacts.OptimType.AdamW, prefix=output_path
    )

    # Multiple shapes inference pass are needed
    inferred_model = shape_inference.infer_shapes(onnx.load(f"{output_path}/training_model.onnx"))
    inferred_model = shape_inference.infer_shapes(inferred_model)
    inferred_model = shape_inference.infer_shapes(inferred_model)

    # Process PoolGrad
    processed_model = process_poolgrad(inferred_model)
    onnx.save(processed_model, f"{output_path}/processed_model.onnx")

    # Process ConvGrad
    processed_model = process_convolution_grad(processed_model)
    onnx.save(processed_model, f"{output_path}/processed_model2.onnx")

    # Process 1D Nodes
    model_simplified, check = simplify(processed_model, skipped_optimizers=["extract_constant_to_initializer"])
    processed_model = process_1d_nodes(model_simplified)
    onnx.save(processed_model, f"{output_path}/processed_model3.onnx")

    # Split Forward, Backward and Optimizer
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(processed_model)
    onnx.utils.extract_model(
        f"{output_path}/processed_model3.onnx",
        f"{output_path}/forward.onnx",
        [element[0] for element in forward_inputs],
        [element[0] for element in forward_outputs],
        True,
    )
    onnx.utils.extract_model(
        f"{output_path}/processed_model3.onnx",
        f"{output_path}/backward.onnx",
        [element[0] for element in backward_inputs],
        [element[0] for element in backward_outputs],
        True,
    )
    onnx.utils.extract_model(
        f"{output_path}/processed_model3.onnx",
        f"{output_path}/forward_backward.onnx",
        [element[0] for element in forward_inputs],
        [element[0] for element in backward_outputs],
        True,
    )


def main():
    # Parse arguments
    args = argparser()

    # Generate ResNet18 ONNX
    os.makedirs(args.output_path, exist_ok=True)
    generate_resnet18_onnx(args.output_path)

    # Load hardware choices from config file
    with open("config.json", "r") as file:
        hw_choices = json.load(file)

    # Generate mapping configuration
    mapping_config = stream_edge_tpu_mapping(4, 4, ["pooling.yaml", "simd.yaml"])

    # Setup processing pool
    max_iter = 200
    pool = multiprocessing.Pool(processes=32)

    # Create generator for configurations
    config_gen = ConfigGenerator(max_iter, hw_choices, mapping_config, None, args.output_path, args.output_path, args.mode)

    # Process configurations in parallel
    results = pool.map(evaluate_performance, config_gen)

    # Save results
    output_file = f"{args.output_path}/results.json"
    with open(output_file, "w") as file:
        json.dump(results, file)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
