"""Builds lesson.ipynb from source.

The notebook is generated, not hand-edited, so its asserts stay in sync with
train.py and the README. The arithmetic and the QLoRA run are inlined so it runs
on Colab with no local files. Regenerate and re-execute with:

    uv run --with nbformat lessons/0010-fine-tuning/build_notebook.py
    uv run --extra notebook python -c "import nbformat, nbclient; \
nb = nbformat.read('lessons/0010-fine-tuning/lesson.ipynb', as_version=4); \
nbclient.NotebookClient(nb, timeout=1200).execute(); \
nbformat.write(nb, 'lessons/0010-fine-tuning/lesson.ipynb')"
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

ASSETS = "https://raw.githubusercontent.com/tamnd/soroban/main/lessons/0010-fine-tuning/assets"
MATHS = "https://github.com/tamnd/soroban/blob/main/maths"

cells.append(md(
f"""# Lesson 0010: fine-tuning arithmetic, LoRA and QLoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tamnd/soroban/blob/main/lessons/0010-fine-tuning/lesson.ipynb)

The ladder ends where most real work starts: taking a model someone else trained and bending it to a new task without paying to train it again. Fine-tuning is an arithmetic problem before it is a machine problem. A full update to a weight matrix is the whole matrix, and for a seven-billion-parameter model that update plus its optimizer state wants about a hundred gigabytes, more than any single consumer card. LoRA replaces each full update with a low-rank stand-in of a few thousand numbers, and QLoRA freezes the base in four-bit precision on top, so the same fine-tune drops to about four gigabytes and fits one 24 GB card. This notebook counts every one of those numbers exactly, then loads Llama-2-7B in four bits and trains it on whatever GPU Colab gives you. The full writeup is in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0010-fine-tuning).

![one weight update: the full square versus B times A]({ASSETS}/factor.png)"""))

cells.append(md(
f"""## 0. The ledger, from one matrix to a memory bill

A full update to a `d x k` weight is `d*k` numbers. LoRA writes it as `B @ A` with `B` of shape `d x r` and `A` of shape `r x k`, so it carries only `r*(d+k)`. `B` starts at zero, so `B @ A` is zero at step zero and the base is untouched until the first gradient. Attached to the seven linears in every Llama-2-7B block, a rank-16 adapter is 39976960 numbers, about 0.59 percent of the 6738415616-parameter model. The full page is [lora.md]({MATHS}/lora.md)."""))

cells.append(code(
'''V, H, I, L = 32000, 4096, 11008, 32
RANK = 16

def lora_params(d, k, r):
    return r * (d + k)

def llama_params():
    emb = V * H
    block = 4*(H*H) + 2*(H*I) + I*H + 2*H  # attn + mlp + two rmsnorms
    return emb + L*block + H + V*H

def adapter_params(r):
    per_block = 4*lora_params(H, H, r) + 2*lora_params(H, I, r) + lora_params(I, H, r)
    return L * per_block

full = H * H
low = lora_params(H, H, RANK)
total = llama_params()
adapt = adapter_params(RANK)
full_ft = total * 16 / 1e9     # ~16 bytes/param: fp16 weight and grad, fp32 Adam and master
qbase = total * 0.5 / 1e9      # 4-bit frozen base, half a byte per parameter
print(f"one 4096x4096 update: full {full}, rank-16 BA {low}, {low/full*100:.2f}%")
print(f"Llama-2-7B {total} params, rank-16 adapter {adapt} = {adapt/total*100:.2f}%")
print(f"full fine-tune {full_ft:.1f} GB, 4-bit base {qbase:.2f} GB")
assert (full, low) == (16777216, 131072)
assert total == 6738415616 and adapt == 39976960
assert abs(full_ft - 107.8) < 0.1 and abs(qbase - 3.37) < 0.01'''))

cells.append(md(
"""## 1. The exit test: a different adapter by hand

Attach a rank-8 LoRA to attention only, the four projections q, k, v, o, in every one of the 32 blocks. Each is 4096-by-4096. Predict the trainable count and its fraction of the base before you run the cell."""))

cells.append(code(
'''attn_only = L * 4 * lora_params(H, H, 8)
print(f"rank-8 attention-only adapter: {attn_only} params, {attn_only/llama_params()*100:.3f}%")
assert attn_only == 8388608'''))

cells.append(md(
f"""## 2. A real QLoRA fine-tune

Load Llama-2-7B in four-bit nf4, attach the rank-16 adapters, and confirm the library reports exactly 39976960 trainable parameters, the hand number to the digit. Then take a few AdamW steps on a five-sentence corpus and watch the loss fall, all on one GPU. This cell downloads about 13 GB the first time and needs a CUDA GPU with bitsandbytes; it skips cleanly without one. On Colab, pick a GPU runtime; a free T4 is enough to load the 4-bit base and train.

![fine-tuning Llama-2-7B: memory against a 24 GB 4090]({ASSETS}/memory.png)"""))

cells.append(code(
'''try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAVE = torch.cuda.is_available()
except ImportError:
    HAVE = False

if not HAVE:
    print("no CUDA GPU with the fine-tuning stack, skipping the run")
    print("Colab: Runtime, Change runtime type, GPU; then run: pip install transformers peft bitsandbytes accelerate")
else:
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    CORPUS = [
        "The soroban ladder ends at the FLOPs ledger and a QLoRA run.",
        "A rank-16 adapter on a 4096 by 4096 matrix trains under one percent of it.",
        "Full fine-tuning a seven billion model needs about one hundred gigabytes.",
        "QLoRA freezes the base in four bits and trains only the small adapters.",
        "The base model is quantized once and never receives a gradient.",
    ]
    model_id = "NousResearch/Llama-2-7b-hf"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="cuda")
    print(f"4-bit base loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB on the GPU")

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=32, target_modules=TARGETS,
                                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable adapter params: {trainable}  (hand count {adapter_params(RANK)})")
    assert trainable == adapter_params(RANK) == 39976960

    tok.pad_token = tok.eos_token
    batch = tok(CORPUS, return_tensors="pt", padding=True).to("cuda")
    labels = batch["input_ids"].clone(); labels[batch["attention_mask"] == 0] = -100
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    first = last = None
    for step in range(41):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 10 == 0: print(f"  step {step:2d}  loss {out.loss.item():.4f}")
        if step == 0: first = out.loss.item()
        last = out.loss.item()
    print(f"loss fell {first:.4f} -> {last:.4f}; peak memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB of 24")
    assert last < first'''))

cells.append(md(
"""## Exercises

1. Attention only across all blocks, rank 16, is `32 * 4 * 16 * (4096+4096) = 16777216` parameters, 0.249 percent of the model. Confirm that is a quarter of the seven-linear adapter, and say why the feed-forward matrices add three quarters.
2. Double the rank to 32. What is the new trainable count, and what fraction of the model? Rank is the one knob: the adapter scales linearly with it.
3. The 4-bit base is 3.37 GB. If you instead loaded the base in 16-bit, how much memory would it take, and would inference alone still fit a 24 GB card? What about a full fine-tune's 107.8 GB?

Worked answers are in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0010-fine-tuning) and asserted in `train.py`. This is the last rung: the ladder from one neuron and a line to a four-bit fine-tune of a seven-billion-parameter model on a single card is complete."""))

nb.cells = cells
nbf.write(nb, str(Path(__file__).parent / "lesson.ipynb"))
print("written")
