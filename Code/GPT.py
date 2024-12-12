import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
block_size = 128
epochs = 5 # Increased for fine-tuning with early stopping
learning_rate = 1e-4  # Reduced learning rate for stability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256  # Reduced embedding size
n_head = 4  # Reduced number of attention heads
n_layer = 4  # Reduced number of layers
dropout = 0.3  # Increased dropout for regularization
use_mixed_precision = True

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
    x, y = x.to(device), y.to(device)
    return x, y

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

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedFoward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = Head(head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model and optimizer
model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)  # Added weight decay
scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Early stopping parameters
patience = 3
best_loss = float('inf')
epochs_without_improvement = 0

# Trackers
train_losses, val_losses = [], []
train_perplexities, val_perplexities = [], []
train_sequence_accuracies, val_sequence_accuracies= [], []
loss_gaps, training_speeds = [], []

# Training loop
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_loss, correct_predictions, total_predictions = 0.0, 0, 0

    for batch_idx in range(len(train_data) // batch_size):
        x, y = get_batch("train")
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            logits, loss = model(x, y)
        if use_mixed_precision:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()

        # Sequence accuracy calculation
        preds = logits.argmax(dim=-1).view(-1)  # Flatten logits to match flattened y
        y_flat = y.view(-1)  # Flatten y to match preds
        correct_predictions += (preds == y_flat).sum().item()  # Compare flattened tensors
        total_predictions += y_flat.numel()  # Total number of elements

    avg_loss = total_loss / (len(train_data) // batch_size)
    train_losses.append(avg_loss)
    train_perplexities.append(torch.exp(torch.tensor(avg_loss)).item())
    train_sequence_accuracies.append(correct_predictions / total_predictions)

    val_loss, val_correct, val_total = 0.0, 0, 0

    model.eval()
    for _ in range(len(val_data) // batch_size):
        x, y = get_batch("val")
        with torch.no_grad():
            logits, loss = model(x, y)
        val_loss += loss.item()

        # Sequence accuracy calculation
        preds = logits.argmax(dim=-1).view(-1)
        y_flat = y.view(-1)
        val_correct += (preds == y_flat).sum().item()
        val_total += y_flat.numel()

    avg_val_loss = val_loss / (len(val_data) // batch_size)
    val_losses.append(avg_val_loss)
    val_perplexities.append(torch.exp(torch.tensor(avg_val_loss)).item())
    val_sequence_accuracies.append(val_correct/val_total)
    loss_gaps.append(avg_val_loss - avg_loss)
    training_speeds.append(time.time() - start_time)

    print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
          f"Train Seq. Accuracy: {train_sequence_accuracies[-1]:.4f}, Val Seq. Accuracies: {val_sequence_accuracies[-1]:.4f}"
          f"Time: {training_speeds[-1]:.2f}s")

    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(drive_model_dir, "best_model.pt"))
        print(f"Model saved to {os.path.join(drive_model_dir, 'best_model.pt')}")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

# Print number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")


# Plot Training and Validation Sequence Accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_sequence_accuracies)), train_sequence_accuracies, label="Training Sequence Accuracies", marker="o", linewidth=2)
plt.plot(range(len(val_sequence_accuracies)), val_sequence_accuracies, label="Validation Sequence Accuracies", linestyle="--", marker="s", linewidth=2)
plt.title("Training and Validation Sequence Accuracies vs Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Plot Training and Validation Perplexity
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_perplexities)), train_perplexities, label="Training Perplexity", marker="o", linewidth=2)
plt.plot(range(len(val_perplexities)), val_perplexities, label="Validation Perplexity", linestyle="--", marker="s", linewidth=2)
plt.title("Training and Validation Perplexity vs Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Perplexity", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label="Training Loss", marker="o", linewidth=2)
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss", linestyle="--", marker="s", linewidth=2)
plt.title("Training and Validation Loss vs Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Generate text
context = torch.tensor([[stoi['H']]], dtype=torch.long).to(device)  # Starting context with character 'H'
generated_tokens = model.generate(context, max_new_tokens=5000)  # Generate 200 new tokens
generated_text = decode(generated_tokens[0].tolist())  # Convert indices to text

# Save generated text to file
output_path = "GPTwithGELU_generated_text.txt"
with open(output_path, "w") as f:
    f.write(generated_text)

print(f"Generated text saved to {output_path}")

# Print the generated text
print("Generated Text:")
print(generated_text)
