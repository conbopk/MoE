import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
import os
from tqdm import tqdm


#Hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400
head_size = 16
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8
top_k = 2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]     # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # decoder: take a list of integers, output a string

#Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.size()
        k = self.key(x)     #(B,T,C)
        q = self.query(x)   #(B,T,C)
        v = self.value(x)   #(B,T,C)

        wei = q @ k.transpose(-2,-1) * c**(-0.5)    #(B,T,T)
        wei = wei.masked_fill_(self.tril[:t,:t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v   #(B,T,C)
        return out

#Multi-Headed Self Attention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)

        #noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

#Now create the sparse mixture of experts module

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.expert = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)     #[batch_size, seq_len, num_expert], [batch_size, seq_len, top_k]
        final_output = torch.zeros_like(x)          #[batch_size, seq_len, n_embed]

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))     #[batch_size * seq_len, n_embed]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))     #[batch_size * seq_len, num_experts]

        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.expert):
            expert_mask = (indices == i).any(dim=-1)    #[batch_size, seq_len]
            flat_mask = expert_mask.view(-1)            #[batch_size * seq_len]
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]      #[num_tokens_for_expert, n_embed]
                expert_output = expert(expert_input)        #[num_tokens_for_expert, n_embed]
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output

class Block(nn.Module):
    """ Mixture of Experts Transformer block: communication followed by computation (multi-head self attention + SparseMoE) """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x

class SparseMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, num_experts=num_experts, top_k=top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

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
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]   #[batch_size, vocab_size]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  #[batch_size,1]
            idx = torch.cat((idx, idx_next), dim=1)     #[batch_size, T+1]
        return idx

def kaiming_init_weights(m):
    if isinstance(m, (nn.Linear)):
        init.kaiming_normal(m.weight)

def save_checkpoint(model, optimizer, loss, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        print(f"Model loaded from {filename}")
        return epoch, loss
    return 0, None

def test_model(model, test_prompt, max_tokens=100):
    model.eval()
    with torch.no_grad():
        # Encode the test prompt
        context = torch.tensor(encode(test_prompt), dtype=torch.long, device=device).unsqueeze(0)

        # Generate text
        generated_text = model.generate(context, max_new_tokens=max_tokens)

        # Decode the generated text
        generated_text = decode(generated_text[0].tolist())

        return generated_text


def main():
    global loss
    model = SparseMoELanguageModel()
    model.apply(kaiming_init_weights)
    model = model.to(device)

    summary(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")

    # Load checkpoint if exists
    start_epoch, loaded_loss = load_checkpoint(model, optimizer, checkpoint_path)

    best_val_loss = float('inf')

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, losses['val'], iter, os.path.join(checkpoint_dir, "best_model.pth"))

        #sample a batch of data
        xb, yb = get_batch('train')

        #evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Lưu checkpoint định kỳ
        if iter % 1000 == 0:
            save_checkpoint(model, optimizer, loss.item(), iter, checkpoint_path)

    # Save last model
    save_checkpoint(model, optimizer, loss.item(), max_iters, os.path.join(checkpoint_dir, 'final_model.pth'))

    test_prompts = [
        "The quick brown fox",
        "In a world where",
        "Once upon a time"
    ]

    print("\nTesting model with sample prompts:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated_text = test_model(model, prompt, max_tokens=50)
        print(f"Generated text: {generated_text}")

if __name__=="__main__":
    main()





















