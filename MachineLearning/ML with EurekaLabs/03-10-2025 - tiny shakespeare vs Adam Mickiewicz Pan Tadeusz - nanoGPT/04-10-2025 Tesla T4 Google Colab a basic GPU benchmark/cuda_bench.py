#!/usr/bin/env python3
"""
cuda_bench.py

A focused CUDA profiling & micro-benchmark script (T4-friendly).
Derived from the provided short train.py benchmark; uses synthetic data,
and does additional profiling/bottleneck checks.

Usage examples:
    # quick 60s profiler run using model.py's GPT if present
    python cuda_bench.py --duration 60 --use-model --profile

    # run synthetic workload (no model required), profile for 60s
    python cuda_bench.py --duration 60 --profile

Key outputs:
 - TensorBoard profiler traces in ./bench_log
 - A printed summary of top CUDA ops and CPU ops
 - GPU memory stats and simple throughput numbers
"""
import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim

# try to import GPT if the user wants to use it
try:
    from model import GPTConfig, GPT  # noqa: E402
    HAVE_GPT = True
except Exception:
    HAVE_GPT = False

# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", help="device to use, e.g. 'cuda' or 'cpu'")
parser.add_argument("--batch-size", type=int, default=12)
parser.add_argument("--block-size", type=int, default=1024)
parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
parser.add_argument("--use-model", action="store_true", help="use model.GPT if available (no checkpoints loaded)")
parser.add_argument("--profile", action="store_true", help="enable torch.profiler run")
parser.add_argument("--duration", type=int, default=60, help="target profiling duration in seconds")
parser.add_argument("--warmup", type=int, default=10, help="warmup iterations before timed runs")
parser.add_argument("--iters-forward", type=int, default=20, help="iterations for forward-only microbenchmark")
parser.add_argument("--iters-backward", type=int, default=20, help="iterations for backward microbenchmark")
parser.add_argument("--compile", action="store_true", help="torch.compile(model) if True")
args = parser.parse_args()
# -------------------------

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu"

# choose dtype heuristically if not provided
if args.dtype is None:
    if device_type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
    elif device_type == "cuda":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
else:
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

print(f"Device: {device}, device_type: {device_type}, dtype: {ptdtype}, use_model_requested: {args.use_model}, have_gpt: {HAVE_GPT}")

# print some device properties (for T4 detection)
if device_type == "cuda":
    try:
        props = torch.cuda.get_device_properties(device)
        print(f"CUDA device name: {props.name}, total_memory: {props.total_memory/1024**3:.2f} GB, major: {props.major}, minor: {props.minor}")
        if "t4" not in props.name.lower() and "tesla-t4" not in props.name.lower():
            print("Warning: device does not look like an NVIDIA T4. The script still runs, but results may differ from a T4 profile.")
    except Exception as e:
        print("Couldn't query CUDA properties:", e)


# Synthetic batch provider (no file I/O)
def make_get_batch(batch_size, block_size, vocab_size=50304, device=device):
    x = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.long, device=device)
    def _get(split="train"):
        # return same tensors (cheap); caller must not mutate them in place
        return x, y
    return _get

get_batch = make_get_batch(args.batch_size, args.block_size, device=device)

# ---------------------------------------------------------------------
# Build workload: either your GPT model (no checkpoint) or a synthetic heavy module
# ---------------------------------------------------------------------
model = None
optimizer = None
use_model = args.use_model and HAVE_GPT

if use_model:
    # small-ish but non-trivial config similar to your base
    gptconf = GPTConfig(
        block_size=args.block_size,
        n_layer=12, n_head=12, n_embd=768,
        dropout=0.0, bias=False,
    )
    model = GPT(gptconf).to(device)
    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    print("Using GPT model for benchmark.")
else:
    # synthetic heavy module: several large matmuls and an attention-like pass
    class SyntheticWorkload(nn.Module):
        def __init__(self, emb, block_size, n_layers=6):
            super().__init__()
            self.emb = emb
            self.block_size = block_size
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(emb, emb, bias=False),
                    nn.GELU(),
                    nn.Linear(emb, emb, bias=False),
                ) for _ in range(n_layers)
            ])
            # a single attention-ish matmul to simulate QK^T and softmax and value matmul
            self.to_q = nn.Linear(emb, emb, bias=False)
            self.to_k = nn.Linear(emb, emb, bias=False)
            self.to_v = nn.Linear(emb, emb, bias=False)

        def forward(self, x_tok):
            # x_tok: [B, L] integer tokens -> create a fake embedding by one-hot matmul (cheap) or embedding lookup
            # We'll use a linear projection on float inputs derived from tokens
            B, L = x_tok.shape
            # make dense "embeddings"
            x = torch.nn.functional.one_hot(x_tok % 32768, num_classes=32768).to(dtype=ptdtype, device=device)
            # reduce dim to emb
            x = x.float() @ torch.randn(x.shape[-1], self.to_q.in_features, device=device, dtype=torch.float32)
            x = x.to(ptdtype)
            # small stack of FFN work
            for layer in self.layers:
                x = layer(x)
            # attention-like
            q = self.to_q(x)  # [B,L,emb]
            k = self.to_k(x)
            v = self.to_v(x)
            # scaled matmul: Q @ K^T
            attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            # reduce to scalar-like loss
            return out.mean()
    # create synthetic module with embedding dim similar to GPT
    emb = 768
    synthetic = SyntheticWorkload(emb, args.block_size, n_layers=6).to(device)
    model = synthetic
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("Using synthetic workload for benchmark (no external model required).")

# helper: single train step (forward+backward+optim)
def train_step(X, Y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits_or_out = model(X, Y) if use_model else model(X)  # GPT returns (logits, loss) in base; synthetic returns scalar
        if use_model:
            # your GPT returns (logits, loss) in base code: handle both shapes
            if isinstance(logits_or_out, tuple) and len(logits_or_out) >= 2:
                loss = logits_or_out[1]
            else:
                # fallback: compute a simple surrogate loss from logits
                logits = logits_or_out
                loss = logits.mean()
        else:
            loss = logits_or_out
    loss.backward()
    optimizer.step()
    # return float
    return loss.detach().item()

def forward_only_step(X, Y):
    model.eval()
    with torch.no_grad(), ctx:
        if use_model:
            logits_or_out = model(X, Y)
            if isinstance(logits_or_out, tuple) and len(logits_or_out) >= 2:
                loss = logits_or_out[1]
            else:
                loss = logits_or_out.mean()
        else:
            loss = model(X)
    return loss.detach().item()

def backward_only_step(X, Y):
    # forward + backward but skip optimizer step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        if use_model:
            logits_or_out = model(X, Y)
            if isinstance(logits_or_out, tuple) and len(logits_or_out) >= 2:
                loss = logits_or_out[1]
            else:
                loss = logits_or_out.mean()
        else:
            loss = model(X)
    loss.backward()
    return loss.detach().item()

# utility to print memory stats
def print_memory_stats(prefix=""):
    if device_type == "cuda":
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        print(f"{prefix} GPU mem allocated: {alloc/1024**2:.1f} MB, reserved: {reserved/1024**2:.1f} MB; peak_alloc: {peak_alloc/1024**2:.1f} MB, peak_reserved: {peak_reserved/1024**2:.1f} MB")
    else:
        print(f"{prefix} (no GPU)")

# -------------------------
# Warmup
# -------------------------
print(f"Warmup: {args.warmup} iterations")
for i in range(args.warmup):
    X, Y = get_batch()
    _ = forward_only_step(X, Y)
    if device_type == "cuda":
        torch.cuda.synchronize()

# -------------------------
# Microbenchmarks: forward-only, backward-only, full step
# -------------------------
def micro_bench(label, fn, iters):
    start = time.time()
    for k in range(iters):
        X, Y = get_batch()
        t0 = time.time()
        val = fn(X, Y)
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        # print a dot occasionally
        if (k+1) % max(1, iters//5) == 0:
            print(f"{label} iter {k+1}/{iters}, sample loss: {val:.4f}, iter_time {(t1-t0)*1000:.2f} ms")
    elapsed = time.time() - start
    tokens = args.batch_size * args.block_size * iters
    print(f"{label}: {iters} iters in {elapsed:.2f}s, tokens/sec: {tokens/elapsed:.1f}")
    print_memory_stats(prefix=f"{label}:")

print("Running microbenchmarks")
micro_bench("Forward-only", forward_only_step, args.iters_forward)
micro_bench("Backward-only", backward_only_step, args.iters_backward)
micro_bench("Full-train-step", train_step, args.iters_backward)

# -------------------------
# Timed profiler run
# -------------------------
if args.profile and device_type == "cuda":
    import torch.profiler
    log_dir = "./bench_log"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Starting torch.profiler for ~{args.duration}s; traces saved to {log_dir}")

    # approximate step loop which will run until duration reached
    start_time = time.time()
    step = 0

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,   # records memory allocs
        with_stack=False,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        # We'll loop until the requested wall-clock duration has passed.
        while True:
            X, Y = get_batch()
            # annotate a region so profiler groups it
            with torch.profiler.record_function("train_step"):
                lossv = train_step(X, Y)
            step += 1
            prof.step()  # mark profiler step
            if device_type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            if step % 10 == 0:
                print(f"Profiler loop step {step}, elapsed {elapsed:.1f}s, last_loss {lossv:.4f}")
            if elapsed >= args.duration:
                break

    # Sync to ensure profiler flush
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed_total = time.time() - start_time
    print(f"Profiler run finished: steps={step}, elapsed={elapsed_total:.2f}s")
    # Print profiler summaries (top CUDA ops and CPU ops)
    try:
        # top CUDA ops
        print("\nTop CUDA ops by cuda_time_total:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print("\nTop CPU ops by cpu_time_total:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    except Exception as e:
        print("Error printing profiler table:", e)

    # additional programmatic analysis: top operators
    try:
        ka = prof.key_averages()
        top_cuda = sorted([k for k in ka], key=lambda k: k.cuda_time_total, reverse=True)[:10]
        print("\nProgrammatic top-10 CUDA ops:")
        for k in top_cuda:
            print(f"  {k.key:40.40s} | cuda_time_total: {k.cuda_time_total:.3f} us | count: {k.count}")
    except Exception:
        pass

    # memory stats
    print_memory_stats(prefix="After profile:")
else:
    if args.profile:
        print("Profiling is requested but no CUDA device detected; skipping torch.profiler section.")

# -------------------------
# Final quick MFU estimate if model has it
# -------------------------
if hasattr(model, "estimate_mfu"):
    # Attempt a quick MFU estimate using the last microbenchmark numbers if possible
    try:
        # approximate steps from earlier microbench
        recent_iters = max(1, args.iters_backward)
        dt = 1.0  # placeholder to avoid division by zero
        # we can't reliably compute dt here for the whole run; use a rough measurement:
        t0 = time.time()
        X, Y = get_batch()
        _ = forward_only_step(X, Y)
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * recent_iters
        mfu = model.estimate_mfu(args.batch_size * 1 * recent_iters, dt)
        print(f"Estimated MFU (rough): {mfu*100:.2f}%")
    except Exception as e:
        print("Could not compute MFU:", e)

print("Done. If you ran with --profile, open TensorBoard on ./bench_log to inspect flame graphs and step traces.")
