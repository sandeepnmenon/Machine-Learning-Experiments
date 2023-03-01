import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyperparameters
batch_size = 64
block_size = 256
epochs = 2000
eval_interval = 500
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_epochs = 200
n_embed=  384
num_heads = 4
num_layers = 3
dropout = 0.2

torch.manual_seed(42)

# Load data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()

# Create vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print('Vocabulary size:', vocab_size)

# Create encoder and decoder
char_to_idx = {ch:i for i,ch in enumerate(vocab)}
idx_to_char = {i:ch for i,ch in enumerate(vocab)}
encode = lambda x: [char_to_idx[ch] for ch in x]
decode = lambda x: ''.join([idx_to_char[idx] for idx in x])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Create batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_epochs)
        for i in range(eval_epochs):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, V = x.shape
        k = self.key(x)  # B,T,head_size
        q = self.query(x)  # B,T,head_size
        v = self.value(x)  # B,T,head_size

        affinity = q @ k.transpose(-2, -1) / V**0.5  # B,T,T
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinity = F.softmax(affinity, dim=-1)
        affinity = self.dropout(affinity)

        # weighted aggregation of values
        out = affinity @ v  # B,T,head_size

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):

        x = self.self_attention(self.layer_norm1(x)) + x
        x = self.ffwd(self.layer_norm2(x)) + x
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embeddings_table = nn.Embedding(block_size, n_embed)
        self.attention_blocks = nn.Sequential(*[Block(n_embed, num_heads=num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(n_embed)
       
        self.ln_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embeddings_table(idx)  # B,T,n_embed
        pos_embedding = self.pos_embeddings_table(torch.arange(T, device=device))  # T,n_embed
        x = token_embedding + pos_embedding
        x = self.attention_blocks(x)
        x = self.layer_norm(x)
        logits = self.ln_head(x)  # B,T,vocab_size
        B, T, V = logits.shape
        
        if targets is None:
            return logits, None
        logits = logits.view(B*T, V)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)  # transpose to get B x T x V
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # take the last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
 
model = BigramLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f'Epoch {epoch}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')


    x_batch, y_batch = get_batch('train')

    # Forward pass
    logits, loss = model(x_batch, y_batch)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Save model
torch.save(model, 'gpt_shapespeare.pt')

# Generate text
context = torch.zeros((1,1), dtype=torch.long).to(device)
generated_output = model.generate(context, max_new_tokens = 2000)
print("Generated text:")
print(decode(generated_output[0].cpu().numpy()))