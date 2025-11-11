import csv
import logging
import os
import shutil
import traceback
from multiprocessing import Pool
from os import getpid
from pathlib import Path
from typing import Literal
import onnx
import onnxruntime as ort
import torch
from onnx import shape_inference
from onnxruntime.training import artifacts
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
from process_onnx import split_forward_backward
from test_ac import apply_onnx_pass, remove_checkpoint
from tools import get_max_offchip_memory, run_stream
import argparse
import random
from deap import base, creator, tools, algorithms
import numpy as np
from typing import Iterable


# -------------------------
# Argument parser
# -------------------------
def argparser():
    parser = argparse.ArgumentParser(description="Stream Hardware Search for ResNet18 (DEAP NSGA-II)")
    parser.add_argument("--output_path", type=str, default="onnx/output/", help="Path to the output directory")
    parser.add_argument("--processes", type=int, required=True, help="number of processes for parallel eval")
    parser.add_argument("--pop_size", type=int, default=20, help="GA population size")
    parser.add_argument("--n_gen", type=int, default=6, help="Number of GA generations")
    parser.add_argument("--cx_prob", type=float, default=0.9, help="Crossover probability (per individual)")
    parser.add_argument("--cx_gene_prob", type=float, default=0.5, help="Binomial crossover per-gene swap prob")
    parser.add_argument(
        "--mut_prob",
        type=float,
        default=0.1,
        help="Mutation probability (per individual triggering bitflip with indpb)",
    )
    parser.add_argument(
        "--mut_indpb", type=float, default=0.1, help="Mutation per-bit probability (when mutation occurs)"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    return parser.parse_args()


# -------------------------
# Helper functions
# -------------------------
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


def bool_list_to_string(bool_list: Iterable[int]) -> str:
    return "".join(["1" if int(b) else "0" for b in bool_list])


# -------------------------
# Evaluator class
# -------------------------
class ActivationCheckpointingEvaluator:
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
        # optimization_vars: dict mapping activation name -> nodes/list
        self.optimization_vars = optimization_vars
        self.model_path = model_path
        self.accelerator_path = accelerator_path
        self.mapping_path = mapping_path
        self.output_path = output_path
        self.forward_inputs = forward_inputs
        self.forward_outputs = forward_outputs
        self.processes = processes
        self.n_var = len(optimization_vars)

    def single_stream_eval(self, x):
        """
        x: iterable of 0/1 or bool values with length == n_var
        returns tuple: (latency, energy, memory_objective)
        """
        pid = getpid()
        folder = f"{self.output_path}{pid}/"
        Path(folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.model_path, f"{folder}model.onnx")

        try:
            recomputations = []
            # optimization_vars is a dict; iterating yields keys in insertion order (py3.7+)
            for variable, activations_key in zip(x, self.optimization_vars, strict=True):
                if bool(variable):
                    recomputations.append([activations_key, self.optimization_vars[activations_key]])
            recomputations.reverse()

            onnx_model = onnx.load(f"{folder}model.onnx")
            checkpointed_model, saved_memory = apply_activation_checkpointing(
                onnx_model, recomputations, self.forward_outputs, self.forward_inputs
            )
            onnx.save(checkpointed_model, f"{folder}checkpointed.onnx")

            processed_model_path, forward_path, backward_pass, opt_pass = apply_onnx_pass(
                output_path=f"{folder}/", model=checkpointed_model
            )

            # Evaluate with Stream (this will run the allocation GA inside Stream)
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

            # Return the same objective orientation as original: minimize latency, energy and maximize saved_memory
            # original used -saved_memory as third objective to minimize (so larger saved_memory is better)
            return latency, energy, -saved_memory

        except Exception as e:
            latency, energy, saved_memory = 1e30, 1e30, 1e30
            error_msg = traceback.format_exc()
            logging.error(f"{e}, trace {error_msg}")
            return latency, energy, -saved_memory


# -------------------------
# Stream evaluation function kept as-is (minor fixes)
# -------------------------
def optimize_allocation_ga_no_id(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]] | None,
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


# -------------------------
# Model generation (same)
# -------------------------
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


# -------------------------
# DEAP utility: binomial-style crossover (per-gene swap probability)
# -------------------------
def cx_binomial(ind1, ind2, gene_swap_prob):
    """Binomial-like crossover: for each gene, with gene_swap_prob swap the genes between parents."""
    size = len(ind1)
    for i in range(size):
        if random.random() < gene_swap_prob:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


# -------------------------
# Main with DEAP NSGA-II (canonical pattern) + stats + halloffame + parallel map
# -------------------------
if __name__ == "__main__":
    args = argparser()
    accelerator_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused_ga_elementwise2.yaml"
    output_path = args.output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)

    optimization_vars, model_path, forward_inputs, forward_outputs = generate_model(output_path)

    ort.set_default_logger_severity(3)
    _logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=_logging_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.CRITICAL)
    error_handler = logging.FileHandler("error_09_11_2025.log")
    error_handler.setLevel(logging.ERROR)
    info_handler = logging.FileHandler("log_09_11_2025.log")
    info_handler.setLevel(logging.INFO)
    warning_handler = logging.FileHandler("warning_09_11_2025.log")
    warning_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().addHandler(error_handler)
    logging.getLogger().addHandler(info_handler)
    logging.getLogger().addHandler(warning_handler)

    # Create evaluator
    evaluator = ActivationCheckpointingEvaluator(
        optimization_vars=optimization_vars,
        forward_inputs=forward_inputs,
        forward_outputs=forward_outputs,
        model_path=model_path,
        accelerator_path=accelerator_path,
        mapping_path=mapping_path,
        output_path=output_path,
        processes=args.processes,
    )

    # -------------------------
    # DEAP setup
    # -------------------------
    POP_SIZE = args.pop_size
    NGEN = args.n_gen
    CX_PROB = args.cx_prob
    CX_GENE_PROB = args.cx_gene_prob
    MUT_PROB = args.mut_prob
    MUT_INDPB = args.mut_indpb
    RANDOM_SEED = args.seed

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    NVAR = evaluator.n_var

    # Fitness: minimization for all 3 objectives => negative weights
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NVAR)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation wrapper
    def deap_eval(individual):
        x = [int(bool(v)) for v in individual]
        return evaluator.single_stream_eval(x)

    toolbox.register("evaluate", deap_eval)

    # Variation operators
    def mate_binomial(ind1, ind2):
        return cx_binomial(ind1, ind2, CX_GENE_PROB)

    toolbox.register("mate", mate_binomial)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUT_INDPB)
    # Selection: tournament DCD for mating, and selNSGA2 for next generation
    toolbox.register("sel_mating", tools.selTournamentDCD)
    toolbox.register("select", tools.selNSGA2)

    # -------------------------
    # Multiprocessing: register pool.map as toolbox.map
    # -------------------------
    with Pool(processes=args.processes) as mp_pool:
        toolbox.register("map", mp_pool.map)

        # Initialize population
        pop = toolbox.population(n=POP_SIZE)

        # Evaluate initial population in parallel
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Hall of Fame and Stats
        hof = tools.HallOfFame(maxsize=POP_SIZE, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        # Register functions to compute min/avg/max per objective
        def arr_min(x):
            return tuple(np.min(np.array(x), axis=0).tolist())

        def arr_avg(x):
            return tuple(np.mean(np.array(x), axis=0).tolist())

        def arr_max(x):
            return tuple(np.max(np.array(x), axis=0).tolist())

        stats.register("min", arr_min)
        stats.register("avg", arr_avg)
        stats.register("max", arr_max)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + stats.fields

        # Record generation 0 stats
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Save history list similarly to your previous approach
        history_list = []
        for j, individual in enumerate(pop):
            history_list.append([0, j] + [int(x) for x in individual] + list(individual.fitness.values))

        # -------------------------
        # Generational loop (canonical NSGA-II pattern)
        # -------------------------
        for gen in range(1, NGEN + 1):
            # Mating selection (tournament DCD)
            mating_parents = toolbox.sel_mating(pop, POP_SIZE)
            offspring = [toolbox.clone(ind) for ind in mating_parents]

            # Variation (crossover then mutation) using tools.varAnd
            offspring = tools.varAnd(offspring, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB)

            # Evaluate invalid individuals in offspring in parallel
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            nevals = 0
            if invalid_ind:
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                nevals = len(invalid_ind)

            # Combine parent and offspring and select next generation via NSGA-II
            pop = toolbox.select(pop + offspring, k=POP_SIZE)

            # Update hall of fame and stats
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=nevals, **record)
            print(f"Generation {gen}: {record}")

            # Append to history_list
            for j, individual in enumerate(pop):
                history_list.append([gen, j] + [int(x) for x in individual] + list(individual.fitness.values))

        # End of pool context (pool will close automatically)

    # -------------------------
    # After GA: extract Pareto front (first nondominated front)
    # -------------------------
    nondom_fronts = tools.sortNondominated(list(pop), k=len(pop))
    first_front = nondom_fronts[0] if len(nondom_fronts) > 0 else []

    best_pop_x = [np.array(ind, dtype=int) for ind in pop]
    best_pop_f = [tuple(ind.fitness.values) for ind in pop]

    pareto_x = [np.array(ind, dtype=int) for ind in first_front]
    pareto_f = [tuple(ind.fitness.values) for ind in first_front]

    # Save results similar to original:
    with open(f"{output_path}result.csv", "w", newline="") as file:
        writer = csv.writer(file)
        if pareto_x:
            writer.writerow(pareto_x[0].tolist())
            writer.writerow(pareto_f[0])
        else:
            writer.writerow([])
            writer.writerow([])
        for i, (ind_x, ind_f) in enumerate(zip(best_pop_x, best_pop_f, strict=True)):
            writer.writerow([i] + ind_x.tolist() + list(ind_f))

    # Save history CSV
    with open(f"{output_path}history.csv", "w", newline="") as file:
        writer = csv.writer(file)
        # header
        writer.writerow(["Run", "Individual"] + [f"X_{i}" for i in range(NVAR)] + ["F1", "F2", "F3"])
        for line in history_list:
            writer.writerow(line)

    # Logbook output saved
    with open(f"{output_path}logbook.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # write header
        header = list(logbook[0].keys())
        writer.writerow(header)
        for entry in logbook:
            writer.writerow([entry[k] for k in header])

    # Try plotting the Pareto front (just a check, not required)
    try:
        import matplotlib.pyplot as plt  # usually available; if not, this block will be skipped

        if pareto_f:
            pareto_arr = np.array(pareto_f)
            # scatter latency vs energy colored by saved_memory (third objective is -saved_memory)
            plt.figure()
            plt.scatter(pareto_arr[:, 0], pareto_arr[:, 1])
            plt.xlabel("Latency")
            plt.ylabel("Energy")
            plt.title("Pareto front (Latency vs Energy)")
            plt.grid(True)
            plt.savefig(f"{output_path}pareto_scatter.png")
            plt.close()
    except Exception:
        logging.warning("matplotlib not available or plotting failed; skipping plot.")

    # Final logs
    logging.critical("Pareto front fitnesses: %s", pareto_f)
    logging.critical("Pareto front individuals: %s", pareto_x)
    # Save hall of fame
    try:
        with open(f"{output_path}hall_of_fame.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for i, ind in enumerate(hof):
                writer.writerow([i] + ind.tolist() + list(ind.fitness.values))
    except Exception:
        logging.warning("Could not save hall of fame.")

    print("Done. Results saved to:", output_path)
