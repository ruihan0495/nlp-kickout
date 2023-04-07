'''
Define all the network modules here
See https://dugas.ch/artificial_curiosity/GPT_architecture.html
'''
import torch
import math
import torch.nn as nn
from einops import rearrange

class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, num_heads, dropout) -> None:
        super().__init__()
        self.model_dim = input_dim
        self.num_heads = num_heads
        self.head_output_dim = self.model_dim/num_heads
        self.dropout = nn.Dropout(p=dropout)
 
        self._Q = nn.Linear(input_dim, self.model_dim, bias=False)
        self._K = nn.Linear(input_dim, self.model_dim, bias=False)
        self._V = nn.Linear(input_dim, self.model_dim, bias=False)

    def forward(self, x):
        # x [B, 2048, 12288]
        qs = rearrange(self._Q(x), 'b s (h he) -> b s h he', h=self.num_heads)
        ks = rearrange(self._K(x), 'b s (h he) -> b s h he', h=self.num_heads)
        vs = rearrange(self._V(x), 'b s (h he) -> b s h he', h=self.num_heads)

        return self._attn_score(qs, ks, vs)
        
    def _attn_score(self, q, k, v):
        q, k, v = q.transpose(2, 1), k.transpose(2, 1), v.transpose(2, 1) # b h s he
        mask = torch.tril(torch.ones((q.shape[-2], q.shape[-2])))
        attn_score_with_mask = (q @ k.transpose(-2, -1)/math.sqrt(self.head_output_dim)).masked_fill(mask == 0, -1e6)
        score = torch.softmax(attn_score_with_mask, dim=-1) 
        score = self.dropout(score) # regularization
        return rearrange(score @ v, 'b h s he -> b s (h he)')
        
class TransformerDecoder(nn.Module):
    '''
    multihead attention -> add and norm -> FFN -> add and norm
    '''
    def __init__(self, input_dim, seq_len, num_heads, dropout) -> None:
        super().__init__()
        self.ffn = nn.Linear(input_dim, input_dim)
        self.ffn_norm = nn.LayerNorm(normalized_shape=[seq_len, input_dim])

        self.attn_norm = nn.LayerNorm(normalized_shape=[seq_len, input_dim])
        self.attention = MultiHeadAttention(input_dim, num_heads, dropout)

    def forward(self, x):
        x = self.attn_norm(self.attention(x) + x)
        x = self.ffn_norm(self.ffn(x) + x)
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class GPTModel(nn.Module):

    def __init__(self, vocab_size, 
                       embedding_size,
                       num_heads,
                       max_seq_len,
                       num_transformer_blocks,
                       dropout,
                       trick_embed=False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformers = [
            TransformerDecoder(embedding_size, max_seq_len, num_heads, dropout) 
                                for _ in range(num_transformer_blocks)
        ]
        self.unembedding = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout) # regularization
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.trick_embed = trick_embed
        
        # tricks
        if self.trick_embed:
            self.emb_norm = nn.LayerNorm(normalized_shape=[max_seq_len, embedding_size])
        
        # init weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            # small init embedding trick
            if self.trick_embed:
                nn.init.uniform_(module.weight, a=1e-4, b=1e-4)

    def forward(self, x):
        x = self.embedding(x)
        if self.trick_embed:
            x = self.emb_norm(x)

        x += self.positional_encoding(x)

        for layer in self.transformers:
            x = layer(x)
 
        x = self.unembedding(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    vocab_size = 11285
    embedding_size = 128
    num_heads = 4
    max_seq_len = 6
    num_transformer_blocks = 48
    dropout = 0.6

    gpt = GPTModel(vocab_size, 
                   embedding_size, 
                   num_heads, 
                   max_seq_len, 
                   num_transformer_blocks, 
                   dropout,
                   True)
    
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    gpt(x)


