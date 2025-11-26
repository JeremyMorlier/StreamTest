import os
import yaml
import numpy as np


def generate_core(xpes: int = 256, ypes: int = 256, buffer_size: int = 268435456, buffer_bandwidth: int = 8192):
    yaml_data = {
        "name": "generic_array",
        "type": "compute",
        "memories": {
            "rf_I": {
                "size": 32,
                "r_cost": 0.2,
                "w_cost": 0.2,
                "area": 0,
                "latency": 1,
                "operands": ["I1"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": 32,
                        "bandwidth_max": 32,
                        "allocation": ["I1, tl"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": 32,
                        "bandwidth_max": 32,
                        "allocation": ["I1, fh"],
                    },
                ],
                "served_dimensions": [],
            },
            "rf_W": {
                "size": 32,
                "r_cost": 0.2,
                "w_cost": 0.2,
                "area": 0,
                "latency": 1,
                "operands": ["I2"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": 4,
                        "bandwidth_max": 32,
                        "allocation": ["I2, tl"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": 4,
                        "bandwidth_max": 32,
                        "allocation": ["I2, fh"],
                    },
                ],
                "served_dimensions": [],
            },
            "rf_O": {
                "size": 32,
                "r_cost": 0.4,
                "w_cost": 0.4,
                "area": 0,
                "latency": 1,
                "operands": ["O"],
                "ports": [
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": 16,
                        "bandwidth_max": 32,
                        "allocation": ["O, fh"],
                    },
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": 16,
                        "bandwidth_max": 32,
                        "allocation": ["O, tl"],
                    },
                    {
                        "name": "w_port_2",
                        "type": "write",
                        "bandwidth_min": 16,
                        "bandwidth_max": 32,
                        "allocation": ["O, fl"],
                    },
                    {
                        "name": "r_port_2",
                        "type": "read",
                        "bandwidth_min": 16,
                        "bandwidth_max": 32,
                        "allocation": ["O, th"],
                    },
                ],
                "served_dimensions": [],
            },
            # Integrate buffer memory into core
            "sram_buffer": {
                "size": buffer_size,
                "r_cost": buffer_bandwidth,
                "w_cost": buffer_bandwidth,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I1, tl", "O, tl", "O, th"],
                    },
                    {
                        "name": "r_port_2",
                        "type": "read",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I2, tl", "O, tl", "O, th"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I1, fh", "O, fh", "O, fl"],
                    },
                    {
                        "name": "w_port_2",
                        "type": "write",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I2, fh", "O, fh", "O, fl"],
                    },
                ],
                "served_dimensions": ["D1", "D2"],
            },
        },
        "operational_array": {
            "unit_energy": 0.04,
            "unit_area": 1,
            "dimensions": ["D1", "D2"],
            "sizes": [xpes, ypes],
        },
    }
    return yaml_data


def generate_simd(npes: int = 64, buffer_size: int = 1048576, buffer_bandwidth: int = 512):
    yaml_data = {
        "name": "simd",
        "type": "compute",
        "memories": {
            "sram_128KB_2rw": {
                "size": 1048576,
                "r_cost": 60,
                "w_cost": 75,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "ports": [
                    {
                        "name": "rw_port_1",
                        "type": "read_write",
                        "bandwidth_min": 512,
                        "bandwidth_max": 512,
                        "allocation": ["I1, fh", "I2, fh", "O, fh", "O, fl"],
                    },
                    {
                        "name": "rw_port_2",
                        "type": "read_write",
                        "bandwidth_min": 512,
                        "bandwidth_max": 512,
                        "allocation": ["I1, tl", "I2, tl", "O, tl", "O, th"],
                    },
                ],
                "served_dimensions": ["D1"],
            },
            "sram_buffer": {
                "size": buffer_size,
                "r_cost": buffer_bandwidth,
                "w_cost": buffer_bandwidth,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "ports": [
                    {
                        "name": "r_port_1",
                        "type": "read",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I1, tl", "O, tl", "O, th"],
                    },
                    {
                        "name": "r_port_2",
                        "type": "read",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I2, tl", "O, tl", "O, th"],
                    },
                    {
                        "name": "w_port_1",
                        "type": "write",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I1, fh", "O, fh", "O, fl"],
                    },
                    {
                        "name": "w_port_2",
                        "type": "write",
                        "bandwidth_min": buffer_bandwidth,
                        "bandwidth_max": buffer_bandwidth,
                        "allocation": ["I2, fh", "O, fh", "O, fl"],
                    },
                ],
                "served_dimensions": ["D1"],
            },
        },
        "operational_array": {
            "unit_energy": 0.1,  # pJ
            "unit_area": 0.01,  # unit
            "dimensions": ["D1"],
            "sizes": [npes],
        },
        "dataflows": {"D1": [f"K, {npes}"]},
    }
    return yaml_data


def generate_offchip():
    yaml_data = {
        "name": "offchip",
        "type": "memory",
        "memories": {
            "dram": {
                "size": 1000000000000,
                "r_cost": 1000,
                "w_cost": 1000,
                "area": 0,
                "latency": 1,
                "operands": ["I1", "I2", "O"],
                "ports": [
                    {
                        "name": "rw_port_1",
                        "type": "read_write",
                        "bandwidth_min": 64,
                        "bandwidth_max": 64,
                        "allocation": [
                            "I1, fh",
                            "I1, tl",
                            "I2, fh",
                            "I2, tl",
                            "O, fh",
                            "O, tl",
                            "O, fl",
                            "O, th",
                        ],
                    }
                ],
                "served_dimensions": ["D1", "D2"],
            }
        },
        "operational_array": {
            "unit_energy": 0,
            "unit_area": 0,
            "dimensions": ["D1", "D2"],
            "sizes": [0, 0],
        },
    }
    return yaml_data


def generate_soc(
    path: str,
    xpes: int = 256,
    ypes: int = 256,
    vector_pes: int = 64,
    buffer_bandwidth: int = 8192,
    buffer_size: int = 268435456,
    off_bandwidth: int = 128,
):
    core_filename = os.path.join(path, "core.yaml")
    simd_filename = os.path.join(path, "simd.yaml")
    offchip_filename = os.path.join(path, "offchip.yaml")
    soc_filename = os.path.join(path, "soc.yaml")

    # Generate the individual component
    to_yaml(generate_core(xpes, ypes, buffer_size, int(buffer_bandwidth / 2)), core_filename)
    to_yaml(generate_simd(vector_pes, buffer_size, int(buffer_bandwidth / 2)), simd_filename)
    to_yaml(generate_offchip(), offchip_filename)

    # Assemble and export the main soc
    yaml_data = {
        "name": "fusemax_like",
        "cores": {
            0: core_filename,
            1: simd_filename,
            2: offchip_filename,
        },
        "offchip_core_id": 2,
        "unit_energy_cost": 0,
        "core_connectivity": [
            {"type": "link", "cores": [0, 1], "bandwidth": int(buffer_bandwidth / 2)},
            {"type": "link", "cores": [0, 2], "bandwidth": off_bandwidth},
            {"type": "link", "cores": [1, 2], "bandwidth": off_bandwidth},
        ],
    }
    to_yaml(yaml_data, soc_filename)
    return yaml_data, soc_filename


def generate_fusemax_mapping(path: str, npes: int = 256):
    mapping_filename = os.path.join(path, "mapping.yaml")
    yaml_data = [
        {
            "name": "default",
            "core_allocation": [0],
            "intra_core_tiling": ["H, all"],
            "inter_core_tiling": [f"D, {npes}"],
        },
        {
            "name": "Conv",
            "core_allocation": [0],
            "intra_core_tiling": ["OY, all"],
            "inter_core_tiling": ["K, 1"],
        },
        {
            "name": "Gemm",
            "core_allocation": [0],
            "intra_core_tiling": ["H, all"],
            "inter_core_tiling": [f"D, {npes}"],
        },
        {
            "name": "Matmul",
            "core_allocation": [0],
            "intra_core_tiling": ["H, all"],
            "inter_core_tiling": [f"D, {npes}"],
        },
        {"name": "Add", "core_allocation": [1], "intra_core_tiling": ["H, all"], "inter_core_tiling": ["B, 1"]},
        {"name": "Mul", "core_allocation": [1], "intra_core_tiling": ["H, all"], "inter_core_tiling": ["B, 1"]},
        {"name": "Div", "core_allocation": [1], "intra_core_tiling": ["H, all"], "inter_core_tiling": ["B, 1"]},
        {"name": "Sqrt", "core_allocation": [1], "intra_core_tiling": ["H, all"], "inter_core_tiling": ["B, 1"]},
        {"name": "Sub", "core_allocation": [1], "intra_core_tiling": ["H, all"], "inter_core_tiling": ["B, 1"]},
    ]
    to_yaml(yaml_data, mapping_filename)
    return yaml_data, mapping_filename


def to_yaml(hardware_architecture, path):
    with open(path, "w") as yaml_file:
        yaml.safe_dump(hardware_architecture, yaml_file, sort_keys=False)
