"""Lesson 0010: fine-tuning arithmetic, LoRA and QLoRA.

Default run asserts every by-hand number and prints the six-line ledger with no
torch. Pass --qlora to load Llama-2-7B in 4-bit, attach LoRA adapters, confirm
the adapter's trainable-parameter count equals the hand number, and train a few
steps to watch the loss fall, all on a single 24 GB GPU.

    python train.py           # numpy-free asserts + headline
    python train.py --qlora   # the real 4-bit run (needs a CUDA GPU + bitsandbytes)
"""

import sys

# Llama-2-7B config: vocabulary, hidden width, MLP intermediate, layers.
V, H, I, L = 32000, 4096, 11008, 32
RANK = 16
# The seven linear layers LoRA is attached to, per transformer block.
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_params(d, k, r):
    """A rank-r update B(d x r) A(r x k) has r(d+k) parameters, against d*k full."""
    return r * (d + k)


def llama_params():
    """Llama-2-7B parameter count, block by block, embedding and head untied."""
    emb = V * H
    block = 4 * (H * H) + 2 * (H * I) + I * H + 2 * H  # attn + mlp + two rmsnorms
    return emb + L * block + H + V * H


def adapter_params(r):
    """LoRA parameters when the seven linears in every block carry a rank-r update."""
    per_block = 4 * lora_params(H, H, r) + 2 * lora_params(H, I, r) + lora_params(I, H, r)
    return L * per_block


def headline():
    """Assert the by-hand numbers, then print the six-line ledger."""
    full = H * H
    lora = lora_params(H, H, RANK)
    assert full == 16777216 and lora == 131072

    total = llama_params()
    assert total == 6738415616, total
    adapt = adapter_params(RANK)
    assert adapt == 39976960, adapt

    full_ft = total * 16       # ~16 bytes/param: fp16 weight and grad, fp32 Adam and master
    qbase = total // 2         # 4-bit frozen base, half a byte per parameter
    qad = adapt * 16
    assert abs(full_ft / 1e9 - 107.8) < 0.1
    assert abs(qbase / 1e9 - 3.37) < 0.01

    print(f"lora   4096x4096 update: full {full}, rank-16 BA = {lora}, {lora/full*100:.2f}% of full")
    print("init   B starts at 0 so BA = 0: the adapter is a no-op at step 0, base unchanged")
    print(f"model  Llama-2-7B {total} params")
    print(f"adapt  lora r=16 on q,k,v,o,gate,up,down = {adapt} trained, {adapt/total*100:.2f}%, rest frozen")
    print(f"memory full fine-tune ~16 bytes/param = {full_ft/1e9:.1f} GB, over a 24 GB 4090")
    print(f"qlora  4-bit base {qbase/1e9:.2f} GB + adapters {qad/1e9:.2f} GB = fits one 4090 with room")


# A tiny corpus with one exact pattern to memorize, so a few steps of fine-tuning
# visibly drop the loss without any dataset download.
CORPUS = [
    "The soroban ladder ends at the FLOPs ledger and a QLoRA run.",
    "A rank-16 adapter on a 4096 by 4096 matrix trains under one percent of it.",
    "Full fine-tuning a seven billion model needs about one hundred gigabytes.",
    "QLoRA freezes the base in four bits and trains only the small adapters.",
    "The base model is quantized once and never receives a gradient.",
]


def qlora():
    """Load Llama-2-7B in 4-bit, attach LoRA, confirm the count, and train briefly."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"missing a dependency ({e.name}), skipping the run; needs torch, "
              "transformers, peft, bitsandbytes, accelerate")
        return
    if not torch.cuda.is_available():
        print("no CUDA GPU found, skipping the QLoRA run")
        return

    model_id = "NousResearch/Llama-2-7b-hf"  # ungated mirror of Llama-2-7B, identical weights
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="cuda")
    base_mem = torch.cuda.memory_allocated() / 1e9
    print(f"4-bit base loaded: {base_mem:.2f} GB on the GPU")

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=RANK, lora_alpha=32, target_modules=TARGETS, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable adapter params: {trainable}  (hand count {adapter_params(RANK)})")
    assert trainable == adapter_params(RANK), (trainable, adapter_params(RANK))
    print(f"that is {trainable/llama_params()*100:.2f}% of the {llama_params()} base parameters")

    # Tokenize the tiny corpus into one batch and take a few optimizer steps.
    tok.pad_token = tok.eos_token
    batch = tok(CORPUS, return_tensors="pt", padding=True).to("cuda")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    first = last = None
    for step in range(41):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"  step {step:2d}  loss {out.loss.item():.4f}")
        if step == 0:
            first = out.loss.item()
        last = out.loss.item()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"loss fell {first:.4f} -> {last:.4f}; peak memory {peak:.2f} GB of 24 GB")
    assert last < first, "the adapter should reduce the loss"
    assert peak < 24.0, f"should fit a 24 GB card, used {peak:.1f}"


def main():
    headline()
    if "--qlora" in sys.argv:
        print()
        qlora()


if __name__ == "__main__":
    main()
