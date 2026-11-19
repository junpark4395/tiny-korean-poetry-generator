import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)  # (B, C, H*D)
        K = self.W_k(x)  # (B, C, H*D)
        V = self.W_v(x)  # (B, C, H*D)

        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
        scores = scores / (D ** 0.5)

        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H*D)
        output = self.W_o(hidden)  # (B, C, E)

        return output


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim)
            for _ in range(n_layer)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size)

        self.embed.weight = self.unembed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        B, C = ids.shape
        device = ids.device

        pos = torch.arange(0, C, dtype=torch.long, device=device)
        emb = self.embed(ids)
        pos_emb = self.pos_embed(pos)
        x = emb + pos_emb

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        logits = self.unembed(x)
        return logits

    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'embed_dim': self.embed_dim,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'ff_dim': self.ff_dim,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model