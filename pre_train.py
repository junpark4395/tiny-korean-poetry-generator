import os
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
learning_rate = 1e-3
max_iters = 38000
model_save_path = './output/model_pretrain.pt'
check_point_path = './output/checkpoint.pt'


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
    ids = np.memmap('./output/train.bin', dtype=np.uint16, mode='r')

    model = GPT(
        vocab_size=vocab_size,
        max_context_len=context_len,
        embed_dim=embed_dim,
        n_head=n_head,
        n_layer=n_layer,
        ff_dim=ff_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    start_step = 0
    if os.path.exists(check_point_path):
        checkpoint = torch.load(check_point_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        print(f"resume from step {start_step}")

    losses = []
    log_interval = 100
    eval_interval = 500

    size = int(len(ids) * 0.99)
    train_ids = ids[:size]
    eval_ids = ids[size:]

    for i in range(start_step, max_iters):
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

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': i+1
            }
            torch.save(checkpoint, check_point_path)

    model.save(model_save_path)
