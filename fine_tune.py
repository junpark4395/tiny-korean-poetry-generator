import torch
import numpy as np
import torch.nn.functional as F
from model import GPT

vocab_size=16000
context_len = 256
embed_dim = 384
n_layer = 6
n_head = 6
batch_size = 32
ff_dim = 4 * embed_dim
device = 'mps'
learning_rate = 1e-4
max_iters = 750
model_save_path = './output/model_pretrain_fine_tune.pt'


def load_data(ids):
    size = len(ids)
    idx_list = torch.randint(0, size - context_len, (batch_size,)).tolist()

    batch_x = np.stack([ids[idx:idx + context_len] for idx in idx_list])
    batch_y = np.stack([ids[idx + 1:idx + context_len + 1] for idx in idx_list])

    batch_x = torch.from_numpy(batch_x.astype(np.int64)).to(device)
    batch_y = torch.from_numpy(batch_y.astype(np.int64)).to(device)

    return batch_x, batch_y

@torch.no_grad()
def estimate_loss(model, eval_ids):
    eval_iters = 50
    eval_losses = []
    for i in range(eval_iters):
        batch_x, batch_y = load_data(eval_ids)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        eval_losses.append(loss.item())

    return sum(eval_losses) / eval_iters

if __name__ == "__main__":
    ids = np.memmap('./output/poem_train.bin', dtype=np.uint16, mode='r')

    model = GPT.load_from('./output/model_pretrain.pt', device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = []
    log_interval = 10
    eval_interval = 50

    size = int(len(ids) * 0.90)
    train_ids = ids[:size]
    eval_ids = ids[size:]

    for i in range(max_iters):
        batch_x, batch_y = load_data(train_ids)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (i + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            print(f"step {i + 1:>6}/{max_iters} | loss {avg_loss:.4f}")

        if (i + 1) % eval_interval == 0:
            eval_avg_loss = estimate_loss(model, eval_ids)
            print(f"step {i + 1:>6}/{max_iters} | eval loss {eval_avg_loss:.4f}")

    model.save(model_save_path)
