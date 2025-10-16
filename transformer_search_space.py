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

from hardware_gen.fusemax_hardware_generator import generate_fusemax_mapping, generate_soc
from model.mini_llm import MiniTransformerLM
from tools import apply_onnx_passes, run_stream

_logging.basicConfig(level=_logging.ERROR)
# Set the logging level to ERROR to suppress warnings
ort.set_default_logger_severity(3)


def argparser():
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


class ConfigGenerator:
    def __init__(self, max_iter, hw_choices, mapping_config, hardware_config, path, forward_path, training_path):
        self.max_iter = max_iter
        self.i = 0
        self.hw_choices = hw_choices

        self.mapping_config = mapping_config
        self.hardware_config = hardware_config
        self.path = path
        self.forward_path = forward_path
        self.training_path = training_path
        self.mode = mode

    def __next__(self):
        config = {}
        if self.i < self.max_iter:
            self.i += 1

            config["hardware_config"] = sample_hardware_configs(self.hw_choices)
            config["mapping_config"] = self.mapping_config
            config["path"] = self.path
            config["mode"] = self.mode
            config["training"] = self.training_path
            config["forward"] = self.forward_path
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

    forward_backward_path = config["training"]
    forward_path = config["forward"]

    # Generate the soc
    soc, soc_yaml_path = generate_soc(
        folder,
        hardware_config["XPEs"],
        hardware_config["YPEs"],
        hardware_config["VectorPEs"],
        hardware_config["BufferBandwidth"],
        hardware_config["BufferSize"],
        hardware_config["OffchipBandwidth"],
    )

    # Generate Hardware and Mapping Config
    _, mapping_path = generate_fusemax_mapping(folder)
    # Copy Necessary Files
    shutil.copyfile(forward_backward_path, f"{folder}/training.onnx")
    shutil.copyfile(forward_path, f"{folder}/forward.onnx")

    result["soc"] = soc

    result["forwardbackward"] = {}
    result["forward"] = {}
    # Evaluate Using Stream
    try:
        scme = optimize_allocation_ga_no_id(
            hardware=soc_yaml_path,
            workload=f"{folder}/training.onnx",
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=4,
            nb_ga_individuals=4,
            output_path=f"{folder}/training",
            skip_if_exists=False,
        )
        result["forwardbackward"]["energy"] = scme.energy
        result["forwardbackward"]["latency"] = scme.latency

        scme = optimize_allocation_ga_no_id(
            hardware=soc_yaml_path,
            workload=f"{folder}/forward.onnx",
            mapping=mapping_path,
            mode=mode,
            layer_stacks=layer_stacks,
            nb_ga_generations=4,
            nb_ga_individuals=4,
            output_path=f"{folder}/forward",
            skip_if_exists=False,
        )
        result["forward"]["energy"] = scme.energy
        result["forward"]["latency"] = scme.latency

    except Exception as e:
        _logging.error(f"Error: {e}")
        print(f"Error: {e}")
        result["forwardbackward"]["energy"] = 0
        result["forwardbackward"]["latency"] = 0

        result["forward"]["energy"] = 0
        result["forward"]["latency"] = 0

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

    onnx_path = os.path.join(folder, "test.onnx")
    infered_path = os.path.join(folder, "inferred.onnx")
    output_path = os.path.join(folder, "output/")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Stream Setup
    mode = "fused"
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    # Example usage of similar to LLama2
    divisor_factor = 8
    vocab_size = int(32000 / divisor_factor)
    max_seq_len = int(2048 / divisor_factor)
    d_model = int(4096 / divisor_factor)
    nhead = int(32 / divisor_factor)
    num_layers = int(32 / divisor_factor)
    dim_feedforward = int(11008 / divisor_factor)

    model = MiniTransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )

    # Dummy input (batch_size=1, seq_len=10)
    dummy_input = torch.randint(0, vocab_size, (100, max_seq_len))

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
    )

    base_model = onnx.load(onnx_path, load_external_data=False)

    inits = base_model.graph.initializer
    requires_grad = []
    # skip the embedding layer
    for init in inits[1:]:
        requires_grad.append(init.name)

    inferred_train_onnx_path4, forward_onnx_path, backward_onnx_path, optimizer_onnx_path = apply_onnx_passes(
        base_model, None, output_path, requires_grad, "onnx", check=False
    )

    # Evaluate using Stream
    hw_choices = {
        "XPEs": [64, 128, 256, 512],
        "YPEs": [64, 128, 256, 512],
        "VectorPEs": [32, 64, 128, 256],
        "BufferBandwidth": [2048, 4096, 8192, 16384],
        "BufferSize": [int(int(element * 1024 * 1024 * 8)) for element in [4, 8, 16, 32]],
        "OffchipBandwidth": [512, 1024, 2048, 4096, 8192],
    }

    num_task = 10000
    num_workers = min(num_task, int(os.cpu_count() / 3) + 1)
    chunksize = math.ceil(num_task / num_workers)

    config_generator = ConfigGenerator(
        num_task, hw_choices, None, None, output_path, inferred_train_onnx_path4, forward_onnx_path
    )
    id = 0

    config_iterator = iter(config_generator)
    with Pool(processes=num_workers) as pool:
        r = pool.map(evaluate_performance, config_iterator, chunksize=chunksize)
        print(r)
    # r = process_map(evaluate_performance, config_iterator, max_workers=num_workers, chunksize=chunksize)
    # print(r)
    # for config in Config_Generator:
