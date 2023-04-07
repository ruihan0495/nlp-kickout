from dataclasses import dataclass
from models import GPTModel

@dataclass
class GPTConfig:
    vocab_size: int = 11285
    embedding_size: int = 128
    num_heads: int = 4
    max_seq_len: int = 512
    num_transformer_blocks: int = 12
    dropout: float = 0.6
    lr: float = 0.001
    step_size: int = 100
    
    # Training tricks
    trick_embed: bool = False # SmallInitEmb, see https://github.com/BlinkDL/SmallInitEmb

    def __post_init__(self):
        self.model = GPTModel(self.vocab_size, 
                              self.embedding_size, 
                              self.num_heads, 
                              self.max_seq_len, 
                              self.num_transformer_blocks, 
                              self.dropout)

