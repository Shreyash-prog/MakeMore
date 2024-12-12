import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64  # How many independent sequences to process in parallel
block_size = 128  # Maximum context length for predictions
epochs = 5
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
output_dir = './model_checkpoints'

os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(1337)

# Load dataset
with open('shakespeare_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Bigram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Focus on last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append next token
        return idx

# Initialize the model and optimizer
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Calculate and print the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Number of Trainable Parameters: {count_parameters(model)}")

# Helper function to calculate perplexity
def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

# Training and validation data for visualization
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []

# Training loop with Perplexity Calculation
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx in range(len(train_data) // batch_size):
        x, y = get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 1000 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

    avg_loss = total_loss / (len(train_data) // batch_size)
    train_losses.append(avg_loss)
    train_perplexities.append(calculate_perplexity(avg_loss).item())
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")

    # Validate the model
    model.eval()
    val_loss = 0.0
    for _ in range(len(val_data) // batch_size):
        x, y = get_batch("val")
        logits, loss = model(x, y)
        val_loss += loss.item()
    avg_val_loss = val_loss / (len(val_data) // batch_size)
    val_losses.append(avg_val_loss)
    val_perplexities.append(calculate_perplexity(avg_val_loss).item())
    print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

    # Save model if it has the best validation loss
    if avg_val_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        best_loss = avg_val_loss

# Visualizations
# 1. Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label="Train Loss", color="blue", linewidth=2, marker="o")
plt.plot(range(epochs), val_losses, label="Validation Loss", color="orange", linewidth=2, linestyle="--", marker="s")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Loss", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(output_path_le, dpi=300)
plt.show()

# 2. Training and Validation Perplexity
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_perplexities, label="Train Perplexity", color="green", linewidth=2, marker="d")
plt.plot(range(epochs), val_perplexities, label="Validation Perplexity", color="red", linewidth=2, linestyle="--", marker="x")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Perplexity", fontsize=12)
plt.title("Training and Validation Perplexity", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(output_path_pe, dpi=300)
plt.show()

# Generate text
context = torch.tensor([[stoi['H']]], dtype=torch.long).to(device)  # Starting context with character 'H'
generated_tokens = model.generate(context, max_new_tokens=5000)  # Generate 200 new tokens
generated_text = decode(generated_tokens[0].tolist())  # Convert indices to text

# Save generated text to file
output_path = "Bigram_generated_text.txt"
with open(output_path, "w") as f:
    f.write(generated_text)

print(f"Generated text saved to {output_path}")

# Print the generated text
print("Generated Text:")
print(generated_text)
