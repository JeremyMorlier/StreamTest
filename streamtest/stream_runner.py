from pathlib import Path

from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.perfetto import convert_scme_to_perfetto_json


def get_max_offchip_memory(scme):
    """
    Recovers the maximum memory attained in offchip memory for a given SCME object.

    Args:
        scme (StreamCostModelEvaluation): An evaluated SCME object.

    Returns:
        float: Maximum offchip memory usage attained (in bytes).
    """
    memory_manager = scme.accelerator.memory_manager
    offchip_core_id = memory_manager.offchip_core_id

    # Find the top instance(s) that correspond to the offchip core
    for top_instance, cores in memory_manager.cores_per_top_instance.items():
        if all(core.id == offchip_core_id for core in cores):
            # Get the memory usage cumsum for this instance
            stored_cumsum = memory_manager.top_instance_stored_cumsum[top_instance]
            # stored_cumsum is expected to be an array-like, with [timestep, bits]
            # Convert bits to bytes (divide by 8)
            stored_bytes = [bits / 8 for _, bits in stored_cumsum]
            return max(stored_bytes)
    # If not found, return None or raise
    return None


def run_stream(
    model_path,
    accelerator_path,
    mapping_path,
    id,
    output_path,
    mode="fused",
    layer_stacks=None,
):
    Path(output_path, str(id)).mkdir(parents=True, exist_ok=True)

    # if layer_stacks is None:
    #     layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
    # Evaluate Using Stream
    # try :
    scme = optimize_allocation_ga(
        hardware=accelerator_path,
        workload=model_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        nb_ga_generations=4,
        nb_ga_individuals=4,
        experiment_id=id,
        output_path=output_path,
        skip_if_exists=False,
    )
    # except Exception as e:
    #     logging.error(f"Error during optimization: {e}")

    # Load in the CostModelEvaluationLUT from the run
    cost_lut_path = f"{output_path}/{id}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)
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

    # Plotting memory usage of best SCME
    scme.plot_memory_usage((0,), (100,), fig_path=f"{output_path}/{id}/memory.png")

    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=f"{output_path}/{id}/scme.json")

    memory = get_max_offchip_memory(scme)
    return scme.latency, scme.energy, memory
