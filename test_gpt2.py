import onnx
import torch

from model.mini_llm import MiniTransformerLM
from tools import apply_onnx_passes, run_stream

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
dummy_input = torch.randint(0, vocab_size, (1, max_seq_len))

output_path = "results/minillm/"
soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mini_transformer_lm.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=16,
)

base_model = onnx.load("mini_transformer_lm.onnx", load_external_data=False)
print(len(base_model.graph.output))

inits = base_model.graph.initializer
requires_grad = []
# skip the embedding layer
for init in inits[1:]:
    # if len(init.dims) != 1 :
    requires_grad.append(init.name)

inferred_train_onnx_path4, _, _, _ = apply_onnx_passes(
    base_model, None, output_path, requires_grad, "onnx", check=False
)
run_stream(inferred_train_onnx_path4, soc_path, mapping_path, id=3, output_path=output_path, mode="fused")
