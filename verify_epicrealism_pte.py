"""Verify out/epicrealism_512_xnnpack_fp32.pte in ExecuTorch runtime:
- load encode / denoise / decode methods
- run the 4-step loop with TCD scheduler (eta=0.3)
- save out/preview_epicrealism_xnnpack_fp32.png
"""
import os
import time
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer

import executorch.backends.xnnpack  # registers XNNPACK delegate
from executorch.runtime import Runtime
from diffusers import TCDScheduler

PTE = "out/epicrealism_512_xnnpack_fp32.pte"
PROMPT = ("portrait photo of a beautiful young woman with long wavy brown hair, "
          "soft gentle smile, natural lips, golden hour lighting, soft bokeh background, "
          "8k uhd, dslr, high quality, film grain, Fujifilm XT3")
SEED = 1234

print("--- Initializing ExecuTorch Runtime ---", flush=True)
rt = Runtime.get()
program = rt.load_program(PTE)
print("Available methods:", sorted(program.method_names), flush=True)

size_mb = os.path.getsize(PTE) / 1024 / 1024
print(f"Total .pte package size: {size_mb:.1f} MB", flush=True)

tok_dir = "models/swmansion_tokenizer"
tok = CLIPTokenizer.from_pretrained(tok_dir)
tokens = tok([PROMPT], max_length=77, padding="max_length", truncation=True,
             return_tensors="pt", return_attention_mask=False).input_ids.to(torch.int64)

enc = program.load_method("encode")
den = program.load_method("denoise")
dec = program.load_method("decode")

sched = TCDScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False,
    clip_sample=False,
)
sched.set_timesteps(4)

g = torch.Generator().manual_seed(SEED)
latents = torch.randn(1, 4, 64, 64, generator=g)

start_time = time.time()
with torch.no_grad():
    # 1. Text encoding
    t0 = time.time()
    embeds = enc.execute((tokens,))[0]
    t_enc = time.time() - t0

    # 2. 4-step Denoising loop
    t0 = time.time()
    g_noise = torch.Generator().manual_seed(SEED)
    for i, t in enumerate(sched.timesteps):
        t_tensor = torch.tensor([t.item()], dtype=torch.int64)
        eps = den.execute((latents, t_tensor, embeds))[0]
        latents = sched.step(eps, t, latents, eta=0.3, generator=g_noise).prev_sample
        print(f"    Step {i+1}/4 (t={t.item()}) completed", flush=True)
    t_den = time.time() - t0

    # 3. Full VAE decoding
    t0 = time.time()
    image = dec.execute((latents,))[0]
    t_dec = time.time() - t0

total_time = time.time() - start_time
print(f"Inference timing: encode={t_enc:.2f}s, 4-step denoise={t_den:.2f}s, decode={t_dec:.2f}s, total={total_time:.2f}s", flush=True)

arr = (image[0].movedim(0, 2).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
out_path = "out/preview_epicrealism_xnnpack_fp32.png"
Image.fromarray(arr).save(out_path)
print(f"Saved {out_path} (shape={arr.shape}, mean={arr.mean():.1f}, std={arr.std():.1f})", flush=True)

if os.path.isfile("out/epicrealism_eager_baseline.png"):
    eager_img = np.asarray(Image.open("out/epicrealism_eager_baseline.png"))
    diff = np.abs(eager_img.astype(np.float32) - arr.astype(np.float32)).mean()
    print(f"Mean pixel diff vs eager FP32 baseline: {diff:.2f}/255")

print("VERIFY_EPICREALISM_FP32_PTE_OK", flush=True)
