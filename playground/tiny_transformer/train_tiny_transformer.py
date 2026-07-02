import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------
# Hyperparameters
# -------------------------

batch_size = 32
block_size = 128          # длина контекста
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100

n_embd = 128              # размер эмбеддинга
n_head = 4                # количество attention heads
n_layer = 4               # количество transformer blocks
dropout = 0.1

torch.manual_seed(42)

# -------------------------
# Data
# -------------------------

with open("war_and_peace.ru.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    source = train_data if split == "train" else val_data

    # случайные стартовые позиции
    ix = torch.randint(len(source) - block_size, (batch_size,))

    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out

# -------------------------
# Model
# -------------------------

class Head(nn.Module):
    """
    One self-attention head.
    """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal mask: токен не должен видеть будущее
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # [B, T, head_size]
        q = self.query(x)    # [B, T, head_size]

        # attention scores
        wei = q @ k.transpose(-2, -1)   # [B, T, T]

        # scaling
        wei = wei * (k.shape[-1] ** -0.5)

        # causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)    # [B, T, head_size]

        out = wei @ v        # [B, T, head_size]

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads in parallel.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    MLP after attention.
    """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block:
    communication + computation.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # pre-norm transformer
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # [B, T, C]

        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)  # [T, C]

        x = tok_emb + pos_emb  # [B, T, C]

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # берем только последние block_size токенов
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # берем logits последней позиции
            logits = logits[:, -1, :]  # [B, vocab_size]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = TinyGPT().to(device)

print(f"Device: {device}")
print(f"Vocab size: {vocab_size}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -------------------------
# Training
# -------------------------

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -------------------------
# Generation
# -------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000)[0].tolist()

print()
print("Generated text:")
print(decode(generated))