import torch
import torch.nn.functional as f
from torch import nn


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask


class NoNormTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        seq_len = src.size(1)
        mask = generate_square_subsequent_mask(seq_len).to(src.device)
        attn_output, _ = self.self_attn(src, src, src, attn_mask=mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)

        # Feedforward
        ff_output = self.linear2(self.dropout(f.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        return src


class NoNormTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size=10000, d_model=128, nhead=4, num_layers=2, max_seq_len=64, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = NoNormTransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward)
        self.transformer = NoNormTransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return f.log_softmax(x, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


if __name__ == "__main__":
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

    dummy_input = torch.randint(0, vocab_size, (1, max_seq_len))
    model(dummy_input)
