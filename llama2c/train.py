#!/usr/bin/env python3
"""
train.py — train llama2.c's Transformer on a LOCAL uint16 token corpus.

Unlike upstream llama2.c's train.py (which streams TinyStories from HuggingFace),
this script reads flat little-endian uint16 arrays from disk — the exact format
that tools/prep_thai.py produces:

    <data_dir>/train.bin    [BOS] tok ... [EOS] [BOS] ... [EOS]  (concatenated)
    <data_dir>/val.bin      same, held-out documents

It saves ckpt.pt in the same layout llama2.c's export.py reads, so afterwards:

    python export.py stories15M_thai_q80.bin --checkpoint out_thai/ckpt.pt --version 2

Apple Silicon (MPS) example:
    python train.py --device=mps --dtype=float32 --max_iters=20000 --lr_decay_iters=20000

Defaults are the 15M "stories15M" architecture (dim=288, 6 layers, 6 heads,
vocab 32000, seq_len 256) so your existing run.c needs no changes.
"""
import os
import time
import math
import argparse

import numpy as np
import torch

from model import Transformer, ModelArgs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--data_dir', default=os.path.join(here, '..', 'data'))
    p.add_argument('--out_dir',  default=os.path.join(here, 'out_thai'))
    p.add_argument('--init_from', default='scratch', choices=['scratch', 'resume'])
    p.add_argument('--finetune', action='store_true',
                   help='with --init_from resume: restart the LR schedule at the resume point')
    # model architecture (keep run.c unchanged)
    p.add_argument('--dim', type=int, default=288)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=6)
    p.add_argument('--n_kv_heads', type=int, default=6)
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--multiple_of', type=int, default=32)
    p.add_argument('--max_seq_len', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.0)
    # optimizer
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad_clip', type=float, default=1.0)
    # lr schedule (cosine)
    p.add_argument('--warmup_iters', type=int, default=1000)
    p.add_argument('--lr_decay_iters', type=int, default=100000)
    p.add_argument('--min_lr', type=float, default=1e-4)
    # training
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--max_iters', type=int, default=100000)
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--eval_iters', type=int, default=100)
    p.add_argument('--log_interval', type=int, default=50)
    # system
    p.add_argument('--device', default='mps')
    p.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'float16': torch.float16,
               'bfloat16': torch.bfloat16}[args.dtype]

    # ---- data: flat uint16 arrays (nanoGPT / prep_thai.py format) ----
    train_path = os.path.join(args.data_dir, 'train.bin')
    val_path = os.path.join(args.data_dir, 'val.bin')
    assert os.path.exists(train_path), f"missing {train_path} — run tools/prep_thai.py first"
    assert os.path.exists(val_path), f"missing {val_path} — run tools/prep_thai.py first"
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - args.max_seq_len, (args.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + args.max_seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.max_seq_len].astype(np.int64)) for i in ix])
        return x.to(args.device), y.to(args.device)

    # ---- model ----
    model_args = dict(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        multiple_of=args.multiple_of,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    model = Transformer(ModelArgs(**model_args)).to(args.device).to(ptdtype)

    iter_num = 0
    best_val_loss = 1e9
    lr_offset = 0
    if args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        iter_num = ckpt.get('iter_num', 0)
        best_val_loss = ckpt.get('best_val_loss', 1e9)
        if args.finetune:
            lr_offset = iter_num
            print(f"resumed from {ckpt_path} at iter {iter_num}  (finetune: LR schedule restarts here)")
        else:
            print(f"resumed from {ckpt_path} at iter {iter_num}")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    # ---- cosine lr schedule ----
    def get_lr(it):
        it -= lr_offset
        if it < args.warmup_iters:
            return args.learning_rate * it / max(1, args.warmup_iters)
        if it > args.lr_decay_iters:
            return args.min_lr
        d = (it - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * d)) * (args.learning_rate - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ('train', 'val'):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                model(X, Y)
                losses[k] = model.last_loss
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- train ----
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M   device: {args.device}   dtype: {args.dtype}")
    print(f"data: {train_path} ({len(train_data):,} tok)  /  {val_path} ({len(val_data):,} tok)")

    model.train()
    t0 = time.time()
    running_loss = None

    for step in range(iter_num, args.max_iters + 1):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if step % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}  lr {lr:.2e}")
            best_val_loss = min(best_val_loss, losses['val'])
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': step,
                'best_val_loss': best_val_loss,
            }, os.path.join(args.out_dir, 'ckpt.pt'))

        # forward / backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            X, Y = get_batch('train')
            model(X, Y)
            (model.last_loss / args.gradient_accumulation_steps).backward()
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_val = model.last_loss.item()  # mean loss of the last micro-batch
        running_loss = loss_val if running_loss is None else 0.9 * running_loss + 0.1 * loss_val
        if step % args.log_interval == 0:
            dt = time.time() - t0
            tok_per_sec = args.batch_size * args.gradient_accumulation_steps * args.max_seq_len * args.log_interval / dt
            print(f"step {step}: loss {running_loss:.4f}  lr {lr:.2e}  {tok_per_sec:.0f} tok/s")
            t0 = time.time()

    print(f"done. checkpoint -> {os.path.join(args.out_dir, 'ckpt.pt')}")


if __name__ == '__main__':
    main()
