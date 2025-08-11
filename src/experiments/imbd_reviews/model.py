import torch
from torch import nn

from src.transformer.encoder import Encoder
from src.transformer.positional_encoding import PositionalEncoding


class IMDbTransformer(nn.Module):
    """Encoder-only Trandsformer model for sentyment recognition IMdB dataset"""
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, h: int, n_layers: int, max_seq_len: int):
        super().__init__()

        self.we = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)

        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, h=h, n=n_layers)
        self.classifier = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        in_emb = self.we(sentence)
        pos_emb = self.pe(in_emb)
        z = self.encoder(in_emb, pos_emb)
        pooled = z.mean(dim=1)
        logits = self.classifier(pooled)
        return self.sigmoid(logits)
