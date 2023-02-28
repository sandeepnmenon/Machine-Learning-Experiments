import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyperparameters
batch_size = 32
block_size = 8
epochs = 5000
eval_interval = 500
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_epochs = 200

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
encode = lambda x: np.array([char_to_idx[ch] for ch in x])
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

# Bigram model

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embeddings(idx)
        B, T, V = logits.shape
        
        if targets is None:
            return logits, None
        logits = logits.view(B*T, V)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)  # transpose to get B x T x V
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # take the last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
 
model = BigramLanguageModel(vocab_size).to(device)

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


# Generate text
context = torch.zeros((1,1), dtype=torch.long).to(device)
generated_output = model.generate(context, max_new_tokens = 500)
print("Generated text:")
print(decode(generated_output[0].cpu().numpy()))