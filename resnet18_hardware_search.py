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

import onnxruntime as ort
import torch
from onnxruntime.training import artifacts
from onnxsim import simplify
from tqdm.contrib.concurrent import process_map
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_load, pickle_save

import onnx
from onnx import shape_inference
from process_onnx import (
    process_1D_nodes,
    process_convGrad,
    process_PoolGrad,
    split_forward_backward,
)
from resnet18 import ResNet18
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
from stream_hardware_generator import (
    stream_edge_tpu,
    stream_edge_tpu_core,
    stream_edge_tpu_mapping,
    to_yaml,
)

_logging.basicConfig(level=_logging.ERROR)
# Set the logging level to ERROR to suppress warnings
ort.set_default_logger_severity(3)


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Stream Hardware Search for ResNet18")
    parser.add_argument("--output_path", type=str, default="onnx/output/", help="Path to the output directory")

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


class Config_Generator:
    def __init__(self, max_iter, hw_choices, mapping_config, hardware_config, path, nn_path):
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
    mapping_config = config["mapping_config"]
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
        "./core.yaml",
        ["./pooling.yaml", "./simd.yaml"],
        "./offchip.yaml",
        32,
        0,
    )
    mapping = stream_edge_tpu_mapping(
        hardware_config["xPE"],
        hardware_config["yPE"],
        ["./pooling.yaml", "./simd.yaml"],
    )
    # Copy Necessary Files
    shutil.copyfile(f"{folder}/../../pooling.yaml", f"{folder}/pooling.yaml")
    shutil.copyfile(f"{folder}/../../simd.yaml", f"{folder}/simd.yaml")
    shutil.copyfile(f"{folder}/../../offchip.yaml", f"{folder}/offchip.yaml")
    shutil.copyfile(forward_backward_path, f"{folder}/forward_backward.onnx")
    shutil.copyfile(forward_path, f"{folder}/forward.onnx")
    shutil.copyfile(backward_path, f"{folder}/backward.onnx")
    to_yaml(core, f"{folder}/core.yaml")
    to_yaml(soc, f"{folder}/hardware_config.yaml")
    to_yaml(mapping, f"{folder}/mapping_config.yaml")
    result["core"] = core
    result["soc"] = soc

    result["forwardbackward"] = {}
    result["forward"] = {}
    result["backward"] = {}
    # Evaluate Using Stream
    try:
        scme = optimize_allocation_ga_no_id(
            hardware=f"{folder}/hardware_config.yaml",
            workload=f"{folder}/forward_backward.onnx",
            mapping=f"{folder}/mapping_config.yaml",
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=4,
            nb_ga_individuals=4,
            output_path=folder,
            skip_if_exists=False,
        )
        result["forwardbackward"]["scme"] = vars(scme)
        result["forwardbackward"]["energy"] = scme.energy
        result["forwardbackward"]["latency"] = scme.latency

        scme = optimize_allocation_ga_no_id(
            hardware=f"{folder}/hardware_config.yaml",
            workload=f"{folder}/forward.onnx",
            mapping=f"{folder}/mapping_config.yaml",
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=4,
            nb_ga_individuals=4,
            output_path=folder,
            skip_if_exists=False,
        )
        result["forward"]["scme"] = vars(scme)
        result["forward"]["energy"] = scme.energy
        result["forward"]["latency"] = scme.latency

        # scme = optimize_allocation_ga(
        #     hardware=f"{folder}/hardware_config.yaml",
        #     workload=f"{folder}/backward.onnx",
        #     mapping=f"{folder}/mapping_config.yaml",
        #     mode=mode,
        #     layer_stacks=layer_stacks,
        #     nb_ga_generations=4,
        #     nb_ga_individuals=4,
        #     experiment_id=id,
        #     output_path=folder,
        #     skip_if_exists=False,
        # )
        # result["backward"]["scme"] = vars(scme)
        # result["backward"]["energy"] = scme["energy"]
        # result["backward"]["latency"] = scme["latency"]
    except Exception as e:
        _logging.error(f"Error: {e}")
        print(f"Error: {e}")
        result["forwardbackward"]["scme"] = 0
        result["forwardbackward"]["energy"] = 0
        result["forwardbackward"]["latency"] = 0

        result["forward"]["scme"] = 0
        result["forward"]["energy"] = 0
        result["forward"]["latency"] = 0

        result["backward"]["scme"] = 0
        result["backward"]["energy"] = 0
        result["backward"]["latency"] = 0

    with open(f"{folder}/resultt.txt", "a") as f:
        json.dump(result, f)
        f.write("\n")
    # break


if __name__ == "__main__":
    args = argparser()
    folder = args.output_path

    logger = _logging.getLogger(__name__)

    _logging.disable(_logging.CRITICAL)
    stream_handler = _logging.StreamHandler()
    stream_handler.setLevel(_logging.CRITICAL)
    logger.addHandler(stream_handler)
    error_handler = _logging.FileHandler("error.log")
    error_handler.setLevel(_logging.ERROR)
    logger.addHandler(error_handler)

    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    train_onnx_path = f"{folder}/training_model.onnx"
    inferred_train_onnx_path = f"{folder}/infered_training_model.onnx"
    inferred_train_onnx_path2 = f"{folder}/infered_training_model2.onnx"
    inferred_train_onnx_path3 = f"{folder}/forward_backward.onnx"

    output_path = f"{folder}/output"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Stream Setups
    mode = "fused"
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    # Copy necessary files for Stream
    shutil.copyfile("stream/stream/inputs/examples/hardware/cores/pooling.yaml", f"{output_path}/pooling.yaml")
    shutil.copyfile("stream/stream/inputs/examples/hardware/cores/simd.yaml", f"{output_path}/simd.yaml")
    shutil.copyfile("stream/stream/inputs/examples/hardware/cores/offchip.yaml", f"{output_path}/offchip.yaml")

    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        # if len(init.dims) != 1 :
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)
    # Now, we can invoke generate_artifacts with this custom loss function
    artifacts.generate_artifacts(
        base_model,
        requires_grad=requires_grad,
        loss=loss,
        optimizer=artifacts.OptimType.AdamW,
        prefix=folder,
    )

    # Infer training graph
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)
    inferred_model = shape_inference.infer_shapes_path(train_onnx_path, inferred_train_onnx_path)

    onnx.save(
        process_convGrad(process_PoolGrad(onnx.load(inferred_train_onnx_path))),
        inferred_train_onnx_path2,
    )
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path2, inferred_train_onnx_path2)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path2, inferred_train_onnx_path2)
    model_simplified, check = simplify(
        onnx.load(inferred_train_onnx_path2),
        skipped_optimizers=["extract_constant_to_initializer"],
    )
    onnx.save(process_1D_nodes(model_simplified), inferred_train_onnx_path3)
    inferred_model = shape_inference.infer_shapes_path(inferred_train_onnx_path3, inferred_train_onnx_path3)
    print(onnx.checker.check_model(inferred_train_onnx_path3))

    # Split Forward and Backward
    onnx_model = onnx.load(inferred_train_onnx_path3)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(onnx_model)
    onnx.utils.extract_model(
        inferred_train_onnx_path3,
        f"{folder}/forward.onnx",
        forward_inputs,
        forward_outputs,
        True,
    )
    onnx.utils.extract_model(
        inferred_train_onnx_path3,
        f"{folder}/backward.onnx",
        backward_inputs,
        backward_outputs,
        True,
    )

    # Evaluate using Stream
    hw_choices = {
        "n_SIMDS": [16, 32, 64, 128],
        "n_computes_lanes": [1, 2, 4, 8],
        "PE_Memory": [int(int(element * 1024 * 1024 * 8)) for element in [0.5, 1, 2, 3, 4]],
        "register_file_size": [int(int(element * 1024 * 8)) for element in [8, 16, 32, 48, 64]],
        "xPE": [1, 2, 4, 6, 8],
        "yPE": [1, 2, 4, 6, 8],
    }

    num_task = 20
    num_workers = min(num_task, int(os.cpu_count() / 2) + 1)
    num_workers = 4
    chunksize = math.ceil(num_task / num_workers)

    config_generator = Config_Generator(num_task, hw_choices, None, None, output_path, nn_path=folder)
    id = 0

    config_iterator = iter(config_generator)
    with Pool(processes=num_workers) as pool:
        r = pool.map(evaluate_performance, config_iterator, chunksize=chunksize)
        print(r)
    # r = process_map(evaluate_performance, config_iterator, max_workers=num_workers, chunksize=chunksize)
    # print(r)
    # for config in Config_Generator:
