import os

import random

import torch
import torch.nn as nn

import onnx
from onnx import shape_inference
from onnxruntime.training import artifacts
import onnxruntime as ort
from onnxsim import simplify
from stream.api import optimize_allocation_ga

import logging

from resnet18 import ResNet18

from stream_hardware_generator import (
    stream_edge_tpu_core,
    stream_edge_tpu,
    stream_edge_tpu_mapping,
    to_yaml,
)
from process_onnx import (
    process_convGrad,
    process_PoolGrad,
    process_1D_nodes,
    split_forward_backward,
)
import json

logging.basicConfig(level=logging.INFO)
# Set the logging level to ERROR to suppress warnings
ort.set_default_logger_severity(3)
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


class Config_Generator:
    def __init__(self, max_iter, hw_choices, mapping_config, hardware_config, path):
        self.max_iter = max_iter
        self.i = 0
        self.hw_choices = hw_choices

        self.mapping_config = mapping_config
        self.hardware_config = hardware_config
        self.path = path
        self.mode = mode

    def __next__(self):
        config = {}
        if self.i < self.max_iter:
            self.i += 1

            config["hardware_config"] = sample_hardware_configs(self.hw_choices)
            config["mapping_config"] = self.mapping_config
            config["path"] = self.path
            config["mode"] = self.mode
            return config
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_iter


if __name__ == "__main__":
    folder = "onnx"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    train_onnx_path = f"{folder}/training_model.onnx"
    inferred_train_onnx_path = f"{folder}/infered_training_model.onnx"
    inferred_train_onnx_path2 = f"{folder}/infered_training_model2.onnx"
    inferred_train_onnx_path3 = f"{folder}/infered_training_model3.onnx"

    output_path = f"{folder}/output"

    # Stream Setups
    core_path = f"{folder}/core.yaml"
    pooling_core_path = os.path.abspath(
        "stream/stream/inputs/examples/hardware/cores/pooling.yaml"
    )
    simd_core_path = os.path.abspath(
        "stream/stream/inputs/examples/hardware/cores/pooling.yaml"
    )
    offchip_core_path = os.path.abspath(
        "stream/stream/inputs/examples/hardware/cores/pooling.yaml"
    )
    mode = "fused"
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list(
        (i,) for i in range(22, 49)
    )

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
    inferred_model = shape_inference.infer_shapes_path(
        train_onnx_path, inferred_train_onnx_path
    )
    inferred_model = shape_inference.infer_shapes_path(
        train_onnx_path, inferred_train_onnx_path
    )
    inferred_model = shape_inference.infer_shapes_path(
        train_onnx_path, inferred_train_onnx_path
    )

    onnx.save(
        process_convGrad(process_PoolGrad(onnx.load(inferred_train_onnx_path))),
        inferred_train_onnx_path2,
    )
    inferred_model = shape_inference.infer_shapes_path(
        inferred_train_onnx_path2, inferred_train_onnx_path2
    )
    inferred_model = shape_inference.infer_shapes_path(
        inferred_train_onnx_path2, inferred_train_onnx_path2
    )
    model_simplified, check = simplify(
        onnx.load(inferred_train_onnx_path2),
        skipped_optimizers=["extract_constant_to_initializer"],
    )
    onnx.save(process_1D_nodes(model_simplified), inferred_train_onnx_path3)
    inferred_model = shape_inference.infer_shapes_path(
        inferred_train_onnx_path3, inferred_train_onnx_path3
    )
    print(onnx.checker.check_model(inferred_train_onnx_path3))

    # Split Forward and Backward
    onnx_model = onnx.load(inferred_train_onnx_path3)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = (
        split_forward_backward(onnx_model)
    )
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
        "PE_Memory": [
            int(int(element * 1024 * 1024 * 8)) for element in [0.5, 1, 2, 3, 4]
        ],
        "register_file_size": [
            int(int(element * 1024 * 8)) for element in [8, 16, 32, 48, 64]
        ],
        "xPE": [1, 2, 4, 6, 8],
        "yPE": [1, 2, 4, 6, 8],
    }
    Config_Generator = Config_Generator(10000, hw_choices, None, None, output_path)

    for config in Config_Generator:
        result = {}
        hardware_config = config["hardware_config"]
        mapping_config = config["mapping_config"]
        mode = config["mode"]

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
            os.path.abspath(core_path),
            [pooling_core_path, simd_core_path],
            offchip_core_path,
            32,
            0,
        )
        mapping = stream_edge_tpu_mapping(
            hardware_config["xPE"],
            hardware_config["yPE"],
            ["pooling.yaml", "simd.yaml"],
        )
        to_yaml(core, core_path)
        to_yaml(soc, f"{folder}/hardware_config.yaml")
        to_yaml(mapping, f"{folder}/mapping_config.yaml")
        result["core"] = core
        result["soc"] = soc


        result["forwardbackward"] = {}
        result["forward"] = {}
        result["backward"] = {}
        # Evaluate Using Stream
        try :
            scme = optimize_allocation_ga(
                hardware=f"{folder}/hardware_config.yaml",
                workload=inferred_train_onnx_path3,
                mapping=f"{folder}/mapping_config.yaml",
                mode=mode,
                layer_stacks=layer_stacks,
                nb_ga_generations=4,
                nb_ga_individuals=4,
                experiment_id=id,
                output_path=output_path,
                skip_if_exists=False,
            )
            result["forwardbackward"]["scme"] = vars(scme)
            result["forwardbackward"]["energy"] = scme["energy"]
            result["forwardbackward"]["latency"] = scme["latency"]
        except Exception as e:
            print(f"Error during forward + backward optimization: {e}")
            result["forwardbackward"]["scme"] = 0
            result["forwardbackward"]["energy"] = 0
            result["forwardbackward"]["latency"] = 0           

        try :
            scme = optimize_allocation_ga(
                hardware=f"{folder}/hardware_config.yaml",
                workload=f"{folder}/forward.onnx",
                mapping=f"{folder}/mapping_config.yaml",
                mode=mode,
                layer_stacks=layer_stacks,
                nb_ga_generations=4,
                nb_ga_individuals=4,
                experiment_id=id,
                output_path=output_path,
                skip_if_exists=False,
            )
            result["forward"]["scme"] = vars(scme)
            result["forward"]["energy"] = scme["energy"]
            result["forward"]["latency"] = scme["latency"]
        except Exception as e:
            print(f"Error during forward optimization: {e}")
            result["forward"]["scme"] = 0
            result["forward"]["energy"] = 0
            result["forward"]["latency"] = 0
            
        try :
            scme = optimize_allocation_ga(
                hardware=f"{folder}/hardware_config.yaml",
                workload=f"{folder}/backward.onnx",
                mapping=f"{folder}/mapping_config.yaml",
                mode=mode,
                layer_stacks=layer_stacks,
                nb_ga_generations=4,
                nb_ga_individuals=4,
                experiment_id=id,
                output_path=output_path,
                skip_if_exists=False,
            )
            result["backward"]["scme"] = vars(scme)
            result["backward"]["energy"] = scme["energy"]
            result["backward"]["latency"] = scme["latency"]
        except Exception as e:
            print(f"Error during backward optimization: {e}")
            result["backward"]["scme"] = 0
            result["backward"]["energy"] = 0
            result["backward"]["latency"] = 0

        with open(f"{folder}/resultt.txt", "a") as f:
            json.dump(result, f)
            f.write("\n")
