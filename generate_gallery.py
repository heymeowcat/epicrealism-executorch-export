"""Generate a showcase gallery from out/epicrealism_512_xnnpack_fp32.pte
using the ExecuTorch C++ runtime (4-step TCD schedule, eta=0.3).

Prompting follows epiCRealism author recommendations (epinikion):
- simple, photographic prompts — no "masterpiece / 8k / best quality" keywords
- simple negative: "cartoon, cgi, render, illustration, painting, drawing"
- low guidance (this export is Hyper-SD 4-step distilled, so CFG must stay low;
  guidance=1.0 disables the negative branch entirely and runs fastest)

Classifier-free guidance is done client-side: when guidance > 1.0, `denoise`
is executed twice per step (positive + negative embeddings) and combined as
    eps = eps_neg + guidance * (eps_pos - eps_neg)

Saves one PNG per entry to out/gallery/.
"""
import os
import time
import torch
from PIL import Image
from transformers import CLIPTokenizer

import executorch.backends.xnnpack  # registers XNNPACK delegate
from executorch.runtime import Runtime
from diffusers import TCDScheduler

PTE = "out/epicrealism_512_xnnpack_fp32.pte"
OUT_DIR = "out/gallery"

NEGATIVE_PROMPT = "cartoon, cgi, render, illustration, painting, drawing"

PROMPTS = [
    ("01_golden_hour_s1234", 1234, 1.0,
     "portrait photo of a young woman with long wavy hair standing in a sunlit meadow at "
     "golden hour, gentle smile, soft backlight shining through her hair, wildflowers"),
    ("02_fisherman_dawn_s42", 42, 1.0,
     "close-up photo of an old fisherman with a weathered face and grey beard standing on a "
     "wooden pier at dawn, damp sea air, soft grey light"),
    ("03_rainy_street_s777", 777, 1.0,
     "street photograph of a woman walking through a rainy city at dusk, wet asphalt "
     "reflecting shop lights, passing cars, shot on Cinestill 800T, red halation around lights"),
    ("04_window_reader_s2024", 2024, 1.0,
     "photo of a young man with curly hair reading a book by a large window in an old "
     "apartment, warm afternoon light, dust in the air, relaxed pose"),
    ("05_snowy_forest_s1997", 1997, 1.0,
     "photo of a woman in a chunky knit sweater walking through a snowy pine forest, "
     "soft diffused winter light, visible cold breath"),
    ("06_gamer_girl_s123", 123, 1.0,
     "photo of a young woman gaming at her desk late at night, colorful RGB keyboard and "
     "monitor glow lighting her face, gaming headset, purple LED strips on the wall behind"),
]

print("--- Initializing ExecuTorch Runtime ---", flush=True)
rt = Runtime.get()
program = rt.load_program(PTE)
enc = program.load_method("encode")
den = program.load_method("denoise")
dec = program.load_method("decode")
print("Methods:", sorted(program.method_names), flush=True)

tok = CLIPTokenizer.from_pretrained("models/swmansion_tokenizer")


def tokenize(text):
    return tok([text], max_length=77, padding="max_length", truncation=True,
               return_tensors="pt", return_attention_mask=False).input_ids.to(torch.int64)


sched = TCDScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False,
    clip_sample=False,
)

os.makedirs(OUT_DIR, exist_ok=True)
for name, seed, guidance, prompt in PROMPTS:
    t_start = time.time()

    # Reset TCD scheduler state (ets history + counter) for every generation
    sched.set_timesteps(4)

    embeds = enc.execute((tokenize(prompt),))[0]
    use_cfg = guidance > 1.0
    if use_cfg:
        neg_embeds = enc.execute((tokenize(NEGATIVE_PROMPT),))[0]

    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 4, 64, 64, generator=g)
    g_noise = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for t in sched.timesteps:
            t_tensor = torch.tensor([t.item()], dtype=torch.int64)
            eps_pos = den.execute((latents, t_tensor, embeds))[0]
            if use_cfg:
                eps_neg = den.execute((latents, t_tensor, neg_embeds))[0]
                eps = eps_neg + guidance * (eps_pos - eps_neg)
            else:
                eps = eps_pos
            latents = sched.step(eps, t, latents, eta=0.3, generator=g_noise).prev_sample

        image = dec.execute((latents,))[0]

    arr = (image[0].movedim(0, 2).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    path = os.path.join(OUT_DIR, f"{name}.png")
    Image.fromarray(arr).save(path)
    print(f"[done] {path}  seed={seed} guidance={guidance}  {time.time() - t_start:.1f}s", flush=True)

print("GENERATE_GALLERY_OK", flush=True)
