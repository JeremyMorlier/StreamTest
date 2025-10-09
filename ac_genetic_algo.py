import shutil
from multiprocessing.pool import ThreadPool
from os import getpid
from pathlib import Path

import onnx
import torch
from onnx import shape_inference
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems import Problem
from pymoo.visualization.scatter import Scatter

from model.resnet18 import ResNet18
from process_onnx import (
    add_optimizer,
    expand_softmax_grad_node,
    process_1d_nodes,
    process_batch_norm,
    process_concat_nodes,
    process_convolution_grad,
    process_poolgrad,
    shape2tuple,
    split_forward_backward,
)
from tools import run_stream
from test_ac import remove_checkpoint, apply_onnx_pass


def apply_activation_checkpointing(model, recomputations, forward_inputs, forward_outputs):
    for recomputation in recomputations:
        model, compute_cost = remove_checkpoint(model, recomputation, forward_inputs, forward_outputs)
    return model


class ActivationCheckpointingProblem(Problem):
    def __init__(
        self, optimization_vars, model_path, accelerator_path, mapping_path, output_path, element_wise_runner=None
    ):
        self.optimization_vars = optimization_vars
        self.model_path = model_path
        self.accelerator_path = accelerator_path
        self.mapping_path = mapping_path
        self.output_path = output_path

        super().__init__(
            n_var=len(optimization_vars),  # Length of binary string
            n_obj=3,  # Number of objectives
            n_constr=0,  # No constraints
            xl=0,  # Lower bound (binary)
            xu=1,  # Upper bound (binary)
            elementwise_evaluation=True,
            element_wise_runner=element_wise_runner,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Get the process ID for tracking
        pid = getpid()
        folder = f"{output_path}{pid}/"
        Path(folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_path, f"{folder}model.onnx")
        # Generate the ONNX based on X
        recomputations = []
        for variable, activations in (x, self.optimization_vars):
            if variable:
                recomputations.append(activations)

        onnx_model = onnx.load(f"{folder}model.onnx")
        checkpointed_model = apply_activation_checkpointing(onnx_model, recomputations)
        onnx.save(checkpointed_model, f"{folder}checkpointed.onnx")
        processed_model_path, forward_path, backward_pass, opt_pass = apply_onnx_pass(
            output_path=f"{folder}/", model=checkpointed_model
        )

        # Evaluate with Stream
        latency, energy = run_stream(processed_model_path, self.accelerator_path, self.mapping_path, x, f"{folder}")
        out["F"] = [latency, energy]
        # the objectives are the energy, latency and memory


def generate_model(output_path):
    model_path = f"{output_path}model.onnx"
    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, model_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes(model_path)
    onnx.save(model, model_path)

    inits = inferred_model.graph.initializer
    requires_grad = []
    for init in inits:
        requires_grad.append(init.name)
    forward_inputs, backward_inputs, forward_outputs, backward_outputs = split_forward_backward(inferred_model)
    # The input and the loss is not an activation to be checkpointed
    optimization_vars = forward_outputs[1:-1]

    return optimization_vars, model_path


if __name__ == "__main__":
    accelerator_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
    output_path = "results/ga_ac/"

    optimization_vars, model_path = generate_model(output_path)
    # initialize the thread pool and create the runner
    n_threads = 4
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # Run the optimization
    problem = ActivationCheckpointingProblem(
        optimization_vars, model_path, accelerator_path, mapping_path, output_path, element_wise_runner=runner
    )

    algorithm = NSGA2(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=BinomialCrossover(n_points=2, prob=0.9),
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 50),  # Number of generations
        seed=1,
        verbose=True,
    )
    pool.close()

    # Plot the Pareto front
    Scatter().add(res.F).show()
