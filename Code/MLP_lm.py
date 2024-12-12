import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyperparameters
BLOCK_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-4
N_EMBD = 64
N_EMBD2 = 128
OUTPUT_DIR = "./model_checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)

# Load the Tiny Shakespeare dataset
with open("shakespeare_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a mapping from characters to integers
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

# Prepare the data
data = torch.tensor(encode(text), dtype=torch.long)
n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))
train_data = data[:n1]
val_data = data[n1:n2]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(len(val_data) // BATCH_SIZE)
        for k in range(len(val_data) // BATCH_SIZE):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.vocab_size = VOCAB_SIZE
        self.wte = nn.Embedding(VOCAB_SIZE + 1, N_EMBD)  # token embeddings table
        self.mlp = nn.Sequential(
            nn.Linear(BLOCK_SIZE * N_EMBD, N_EMBD2),
            nn.Tanh(),
            nn.Linear(N_EMBD2, VOCAB_SIZE),
        )

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# Initialize the model and optimizer
model = MLP().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_idx in range(len(train_data) // BATCH_SIZE):
        x, y = get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 1000 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

    avg_loss = total_loss / (len(train_data) // BATCH_SIZE)
    train_losses.append(avg_loss)
    train_perplexities.append(calculate_perplexity(avg_loss).item())
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")

    # Validate the model
    model.eval()
    val_loss = 0.0
    for _ in range(len(val_data) // BATCH_SIZE):
        x, y = get_batch("val")
        logits, loss = model(x, y)
        val_loss += loss.item()
    avg_val_loss = val_loss / (len(val_data) // BATCH_SIZE)
    val_losses.append(avg_val_loss)
    val_perplexities.append(calculate_perplexity(avg_val_loss).item())
    print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

    # Save model if it has the best validation loss
    if avg_val_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
        best_loss = avg_val_loss

# Generate from the model
@torch.no_grad()
def generate(model, context, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = context[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
    return context

# Visualizations
# 1. Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(EPOCHS), train_losses, label="Train Loss", color="blue", linewidth=2, marker="o")
plt.plot(range(EPOCHS), val_losses, label="Validation Loss", color="orange", linewidth=2, linestyle="--", marker="s")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Loss", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(output_path_le, dpi=300)
plt.show()

# 2. Training and Validation Perplexity
plt.figure(figsize=(10, 6))
plt.plot(range(EPOCHS), train_perplexities, label="Train Perplexity", color="green", linewidth=2, marker="d")
plt.plot(range(EPOCHS), val_perplexities, label="Validation Perplexity", color="red", linewidth=2, linestyle="--", marker="x")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Perplexity", fontsize=12)
plt.title("Training and Validation Perplexity", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(output_path_pe, dpi=300)
plt.show()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
print(decode(generate(model, context, max_new_tokens=2000)[0].tolist()))
generated_text = decode(generate(model, context, max_new_tokens=2000)[0].tolist())

# Save generated text to file
output_path = "/content/drive/MyDrive/MLP_generated_text.txt"
with open(output_path, "w") as f:
    f.write(generated_text)

print(f"Generated text saved to {output_path}")
