import pickle
import torch
import torch.nn.functional as F
from model import GPT
from tokenizer import Tokenizer


device = 'mps'
model_path = './output/model_pretrain_fine_tune.pt'
tokenizer_path = './output/bpe_rules_16k.pkl'
max_new_tokens = 250
temperature = 0.8

@torch.no_grad()
def generate(model, prompt, tokenizer, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 평가 모드


    device = next(model.parameters()).device  # 모델 파라미터의 디바이스를 가져옴
    ids = torch.tensor([prompt], dtype=torch.long, device=device)

    generated_ids = ids.clone()

    for _ in range(max_new_tokens):
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        logits = model(ids)[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.end_of_text_id:
            break

        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

if __name__ == "__main__":
    rules = None
    with open("output/bpe_rules_16k.pkl", "rb") as f:
        rules = pickle.load(f)

    tokenizer = Tokenizer(rules)
    model = GPT.load_from(model_path, device=device)

    prompt = '여름'
    prompt = tokenizer.encode(prompt)

    for i in range(5):
        print(f"--- sample {i+1} ---")
        generated_text = generate(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        print(generated_text)
        print()