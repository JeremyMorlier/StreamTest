# Generate Hardware accelerator yamls
import yaml

def stream_edge_tpu_core(n_SIMDS, n_compute_lanes, PE_memory, register_file_size) :
    hardware_architecture = {
        "name": "edge_tpu_like_core",
        "memories": {
            "rf_128B": {
                "size": register_file_size,
                "r_bw": 8,
                "w_bw": 8,
                "r_cost": 0.095,
                "w_cost": 0.095,
                "area": 0,
                "r_port": 1,
                "w_port": 1,
                "rw_port": 0,
                "latency": 1,
                "auto_cost_extraction": False,
                "operands": ["I2"],
                "ports": [{"fh": "w_port_1",  "tl": "r_port_1"}],
                "served_dimensions" : [],
            },
            "rf_2B": {
                "size": 16,
                "r_bw": 16,
                "w_bw": 16,
                "r_cost": 0.021,
                "w_cost": 0.021,
                "area": 0,
                "r_port": 2,
                "w_port": 2,
                "rw_port": 0,
                "latency": 1,
                "operands": ["O"],
                "ports":[{"fh": "w_port_1",
                    "tl": "r_port_1",
                    "fl": "w_port_2",
                    "th": "r_port_2",}],
                "served_dimensions": ["D2"]
            },
            "sram_2MB": {
                "size": PE_memory,
                "r_bw": 2048,
                "w_bw": 2048,
                "r_cost": 416.16,
                "w_cost": 378.4,
                "area": 0,
                "r_port": 1,
                "w_port": 1,
                "rw_port": 0,
                "latency": 1,
                "min_r_granularity": 64,
                "min_w_granularity": 64,
                "operands": ["I1", "I2", "O"],
                "ports":[{"fh": "w_port_1", "tl": "r_port_1",}, 
                    {"fh": "w_port_1", "tl": "r_port_1",}, 
                    {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1",}],
                "served_dimensions": ["D1", "D2"],
            },

        },
        "operational_array" : {
            "unit_energy": 0.04,
            "unit_area": 1,
            "dimensions": ["D1", "D2"],
            "sizes": [n_SIMDS, n_compute_lanes],
        },
        "dataflows" : {
            "D1": ["K, 32"],
            "D2": ["C, 32"]
        },
    }
    return hardware_architecture

def stream_edge_tpu_mapping(xPEs, yPEs, additional_cores) :

    pooling_cores = []
    adder_cores = []
    for i, element in enumerate(additional_cores) :
        if "pooling" in element :
            pooling_cores.append(i + (xPEs*yPEs-1))
        if "simd" in element :
            adder_cores.append(i + (xPEs*yPEs-1))
                    
    mapping_config = [
        {"name": "default",
        "core_allocation": [element for element in range(0, xPEs*yPEs)],
        "intra_core_tiling": ["D, all"],
        "inter_core_tiling": ["K, *"],
        },

        {"name": "Conv",
        "core_allocation": [element for element in range(0, xPEs*yPEs)],
        "intra_core_tiling": ["OY, all"],
        "inter_core_tiling": ["K, *"],
        },
        {"name": "Gemm",
        "core_allocation": [element for element in range(0, xPEs*yPEs)],
        "intra_core_tiling": ["D, all"],
        "inter_core_tiling": ["H, *"]},

        {"name": "Pool",
        "core_allocation": pooling_cores,
        "intra_core_tiling": ["OY, all"],
        "inter_core_tiling": ["K, *"],},

        {"name": "MaxPool",
        "core_allocation": pooling_cores,
        "intra_core_tiling": ["OY, all"],
        "inter_core_tiling": ["K, *"]},

        {"name": "AveragePool",
        "core_allocation": pooling_cores,
        "intra_core_tiling": ["OY, all"],
        "inter_core_tiling":["K, *"]},
        {
        "name": "GlobalAveragePool",
        "core_allocation": pooling_cores,
        "intra_core_tiling": ["OY, all"],
        "inter_core_tiling": ["K, *"]},

        {"name": "Add",
        "core_allocation": adder_cores,
        "intra_core_tiling": ["D, all"],
        "inter_core_tiling": ["H, *"]},

    ]
    return mapping_config
def stream_edge_tpu(xPEs, yPEs, core, additional_cores, offchip_core, bandwith, unit_energy_cost) :


    hardware_architecture = {
        "name":"edge_tpu_like",
        "cores": {

        },
        "core_connectivity": [

        ],
        "bandwidth": bandwith,
        "unit_energy_cost": unit_energy_cost,
    }

    # Create Cores
    for yPE in range(0, yPEs) :
        for xPE in range(0, xPEs) :
            hardware_architecture["cores"][xPE + yPE*xPEs] = core

    if additional_cores is not None :
        i =  yPEs*xPEs
        for additional_core in additional_cores :
            hardware_architecture["cores"][i] = additional_core
            i += 1
    if offchip_core is not None :
        hardware_architecture["offchip_core"] = offchip_core
    
    # Connect cores
    for yPE in range(0, yPEs-1) :
        for xPE in range(0, xPEs-1) :
            hardware_architecture["core_connectivity"].append(f"{xPE + yPE*xPEs}, {(xPE + 1) + yPE*xPEs}")
            hardware_architecture["core_connectivity"].append(f"{xPE + yPE*xPEs}, {xPE + (yPE+1)*xPEs}")

    # End of rows and columns
    for xPE in range(0, xPEs-1) :
        hardware_architecture["core_connectivity"].append(f"{xPE + (yPEs - 1)*xPEs}, {(xPE + 1) + (yPEs-1)*xPEs}")
    for yPE in range(0, yPEs -1) :
        hardware_architecture["core_connectivity"].append(f"{(xPEs - 1) + yPE*xPEs}, {(xPEs - 1) + (yPE+1)*xPEs}")

    if additional_cores is not None :
        for yPE in range(0, yPEs) :
            for xPE in range(0, xPEs) :
                i = yPEs*xPEs
                for additional_core in additional_cores :
                    hardware_architecture["core_connectivity"].append(f"{xPE + yPE*xPEs}, {i}")
                    i += 1
    return hardware_architecture

def to_yaml(hardware_architecture, path) :
    with open(path, "w") as yaml_file:
        yaml.safe_dump(hardware_architecture, yaml_file, sort_keys=False)

if __name__ == "__main__" :
    to_yaml(stream_edge_tpu(4, 4, "tpu_like.yaml", ["pooling.yaml", "simd.yaml"], "offchip.yaml", 32, 0), "test.yaml")