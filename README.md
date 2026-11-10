---
license: openrail++
pipeline_tag: text-to-image
tags:
- text-to-image
- stable-diffusion
- executorch
- react-native-executorch
- photorealism
library_name: executorch
---

# epicrealism-512-executorch

This repository hosts the **epicrealism-512** photorealistic portrait models exported for ExecuTorch and the [React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library as ExecuTorch `.pte` programs, ready to run on device.

- **Upstream Base Model**: [emilianJR/epiCRealism (newEra)](https://huggingface.co/emilianJR/epiCRealism)
- **Distillation LoRA**: [ByteDance/Hyper-SD (Hyper-SD15-4steps-lora)](https://huggingface.co/ByteDance/Hyper-SD)
- **Latent Decoder**: [Stability AI AutoencoderKL](https://huggingface.co/stabilityai/sd-vae-ft-mse) (`post_quant_conv` + `decoder` + `scale = 0.18215`)
- **Tokenizer**: [Software Mansion RN ExecuTorch SDXS export](https://huggingface.co/software-mansion/react-native-executorch-sdxs-512-dreamshaper) (CLIP ViT-L/14 `tokenizer.json`)
- **Maintainer**: [@heymeowcat](https://huggingface.co/heymeowcat)

> **Repository layout**: The GitHub repository contains the export pipeline, verification tooling, React Native integration code and documentation. The multi-gigabyte `.pte` binaries are hosted on [Hugging Face](https://huggingface.co/heymeowcat/epicrealism-512-executorch) (GitHub does not host files >100 MB without paid LFS quota).

---

## Download

```bash
# Option A — Hugging Face CLI
hf download heymeowcat/epicrealism-512-executorch \
  --include "xnnpack/*" "tokenizer.json" \
  --local-dir .

# Option B — Direct URLs
BASE="https://huggingface.co/heymeowcat/epicrealism-512-executorch/resolve/main"
curl -L -o epicrealism_512_xnnpack_fp32.pte "$BASE/xnnpack/epicrealism_512_xnnpack_fp32.pte"
curl -L -o tokenizer.json                   "$BASE/tokenizer.json"
```

Or from Python:

```python
from huggingface_hub import hf_hub_download
pte_path = hf_hub_download("heymeowcat/epicrealism-512-executorch",
                           "xnnpack/epicrealism_512_xnnpack_fp32.pte")
```

---

## Variants

| Path (HF repo) | Backend | Precision | Size | Status |
| :--- | :--- | :--- | :--- | :--- |
| `xnnpack/epicrealism_512_xnnpack_fp32.pte` | xnnpack | fp32 | 3,937.9 MB | **Production / Verified** |

---

## Photographic Quality Benchmark

Unlike distilled 1-step models or tiny approximation decoders (such as TAESD) which introduce heavy watercolor smudges and cartoonish skin textures, this model combines **epiCRealism**, **4-step Hyper-SD distillation**, and the **Full AutoencoderKL Decoder** in unquantized **FP32**:

| Implementation | Decoder | Distillation | Visual Quality | Mean Pixel Diff vs Eager |
| :--- | :--- | :--- | :--- | :--- |
| **ExecuTorch `.pte` (Ours)** | **Full AutoencoderKL** | **Hyper-SD 4-Step** | **DSLR Photorealism** (Pores, iris detail, natural hair) | **20.81 / 255 (Parity)** |
| Previous Fast Tier | TAESD (5MB) | 1-Step | Painterly / watercolor smudges | > 42.4 / 255 |

### Sample Output (ExecuTorch Runtime)

Synthesized via ExecuTorch C++ runtime on macOS CPU using the 4-step schedule `[999, 759, 499, 259]` and seed `1234`:

> *"portrait photo of a beautiful young woman with long wavy brown hair, soft gentle smile, natural lips, golden hour lighting, soft bokeh background, 8k uhd, dslr, high quality, film grain, Fujifilm XT3"*

| ExecuTorch Runtime | Eager PyTorch Baseline |
| :---: | :---: |
| ![ExecuTorch output](out/preview_epicrealism_xnnpack_fp32.png) | ![Eager baseline](out/epicrealism_eager_baseline.png) |
| `out/preview_epicrealism_xnnpack_fp32.png` | `out/epicrealism_eager_baseline.png` |

---

## Example Gallery

All images below were generated **on-device style** — straight from the exported `epicrealism_512_xnnpack_fp32.pte` through the ExecuTorch runtime (4-step TCD schedule `[999, 759, 499, 259]`, η = 0.3, guidance 1.0 — the distilled default), using simple photographic prompts following the [epiCRealism author's prompting advice](https://civitai.com/models/25694/epicrealism) (no quality-keyword spam). Negative prompts and classifier-free guidance are supported client-side — see the API contract below. Fully reproducible via [`generate_gallery.py`](generate_gallery.py).

| | |
| :---: | :---: |
| ![Golden hour portrait](out/gallery/01_golden_hour_s1234.png) | ![Old fisherman at dawn](out/gallery/02_fisherman_dawn_s42.png) |
| *"…young woman … in a sunlit meadow at golden hour…"* <br> `seed=1234` | *"…old fisherman with a weathered face … pier at dawn…"* <br> `seed=42` |
| ![Rainy street](out/gallery/03_rainy_street_s777.png) | ![Window reader](out/gallery/04_window_reader_s2024.png) |
| *"…woman walking through a rainy city at dusk, shot on Cinestill 800T…"* <br> `seed=777` | *"…young man with curly hair reading a book by a large window…"* <br> `seed=2024` |
| ![Snowy forest](out/gallery/05_snowy_forest_s1997.png) | ![Gamer girl at RGB desk](out/gallery/06_gamer_girl_s123.png) |
| *"…woman in a chunky knit sweater … snowy pine forest…"* <br> `seed=1997` | *"…young woman gaming at her desk late at night, RGB keyboard and monitor glow…"* <br> `seed=123` |

---

## Inference Benchmarks

Evaluated with ExecuTorch C++ runtime on macOS CPU (Apple Silicon):

| Pipeline Stage | Method Name | Input Shape | Latency (CPU) |
| :--- | :--- | :--- | :--- |
| Text Encoding | `encode` | `int64[1, 77]` | **0.14 s** |
| 4-Step Denoising | `denoise` | `float32[1, 4, 64, 64]` × 4 | **21.52 s** (~5.3 s / step) |
| Latent Decoding | `decode` | `float32[1, 4, 64, 64]` | **11.59 s** |
| **Total Generation** | | | **33.25 s** |

*(Note: On mobile devices with multi-core XNNPACK threading or NPU acceleration, denoising is expected to run substantially faster).*

---

## Prompting Tips

This checkpoint responds best to **simple, natural-language photography prompts**. The epiCRealism author ([epinikion](https://civitai.com/models/25694/epicrealism)) recommends:

- Write a plain description of the scene — keywords like *"masterpiece, best quality, 8k, intricate, highly detailed"* produce **no appreciable change**
- Use a simple negative prompt: `cartoon, cgi, render, illustration, painting, drawing`
- Avoid `cinematic` if you want a natural look, and avoid `1girl` (pushes toward anime/render style)
- Don't over-describe the face — light, shadow and skin detail are already excellent
- Naming a **film stock** (`Cinestill 800T`, `Kodak Portra 400`, `Tri-X 400`) is the most effective realism trick — it steers color, grain and halation better than quality keywords (see gallery image 03)
- Keep CFG low (this export is 4-step distilled: use $w \le 2.5$, see the CFG section above)

---

## Repository Structure

```
epicrealism-executorch-export/                   # GitHub repository (code + docs)
├── README.md                                    # Hugging Face Model Card / Documentation
├── LICENSE                                      # CreativeML Open RAIL++-M License
├── config.json                                  # RN ExecuTorch model descriptor
├── requirements.txt                             # Python dependencies for export & verification
├── download_epic.sh                             # Weights + tokenizer download script
├── export_epicrealism_pte.py                    # PyTorch export & ExecuTorch lowering pipeline
├── verify_epicrealism_pte.py                    # ExecuTorch C++ runtime verification script
├── generate_gallery.py                          # Showcase image generation from the .pte
├── integration_snippet.tsx                      # React Native (useExecutorchModule + Skia)
├── xnnpack/
│   └── config.json                              # Spec-conformant API contract for the .pte
├── models/                                      # gitignored — fetched via ./download_epic.sh
│   ├── epicrealism/epicrealism_newEra.safetensors      # 2.03 GB (source checkpoint)
│   ├── hyper_sd/Hyper-SD15-4steps-lora.safetensors     # 256 MB (distillation LoRA)
│   └── swmansion_tokenizer/tokenizer.json              # CLIP ViT-L/14 tokenizer
└── out/
    ├── epicrealism_512_xnnpack_fp32.pte         # gitignored — 3,937.9 MB, hosted on Hugging Face
    ├── preview_epicrealism_xnnpack_fp32.png     # ExecuTorch Runtime Generated Photo
    ├── epicrealism_eager_baseline.png           # Eager PyTorch Reference Photo
    └── gallery/                                 # Multi-prompt showcase images (generate_gallery.py)
```

The Hugging Face repository contains the release artifacts, following the [react-native-executorch-spec](https://huggingface.co/software-mansion/react-native-executorch-spec) layout:

```
heymeowcat/epicrealism-512-executorch/
├── README.md                                    # Model card (this file)
├── LICENSE                                      # CreativeML Open RAIL++-M License
├── config.json                                  # RN ExecuTorch model descriptor
├── tokenizer.json                               # CLIP ViT-L/14 tokenizer
├── tokenizer_config.json                        # CLIPTokenizer config
├── xnnpack/
│   ├── config.json                              # Spec-conformant API contract (validated)
│   └── epicrealism_512_xnnpack_fp32.pte         # 3,937.9 MB production binary
└── out/
    ├── preview_epicrealism_xnnpack_fp32.png
    ├── epicrealism_eager_baseline.png
    └── gallery/                                 # Showcase images (rendered in the model card)
```

---

## Compatibility

These files are exported for the **ExecuTorch v1.4.1+** runtime using the **XNNPACK Floating Point Partitioner**.

To load these files in React Native ExecuTorch, use the `useExecutorchModule` hook from [`react-native-executorch`](https://www.npmjs.com/package/react-native-executorch).

---

## Model Details & API Contract

The single `.pte` program defines three distinct methods. The diffusion loop is orchestrated by the client:

### 1. `encode`
Computes CLIP prompt embeddings from tokenized prompt IDs:
- **Inputs**: `tokens: int64[1, 77]`
- **Outputs**: `prompt_embeds: float32[1, 77, 768]`
- **Backend**: XNNPACK FP32

### 2. `denoise`
Computes the noise prediction $\epsilon_\theta(z_t, t, c)$ at each timestep:
- **Inputs**:
  1. `latents: float32[1, 4, 64, 64]`
  2. `timestep: int64[1]` (one of `[999, 759, 499, 259]`)
  3. `embeds: float32[1, 77, 768]`
- **Outputs**: `noise_pred: float32[1, 4, 64, 64]`
- **Backend**: XNNPACK FP32

### 3. `decode`
Reconstructs the 512×512 RGB image using the full AutoencoderKL decoder with proper color channel projection:
- **Inputs**: `latents: float32[1, 4, 64, 64]`
- **Internal Formula**:
  $$\text{image} = \text{clamp}\left(\frac{\text{decoder}(\text{post\_quant\_conv}(\text{latents} / 0.18215))}{2} + 0.5, 0.0, 1.0\right)$$
- **Outputs**: `image: float32[1, 3, 512, 512]` in range $[0.0, 1.0]$
- **Backend**: XNNPACK FP32

### Optional: Negative Prompt via Client-Side CFG

The exported program intentionally contains **no classifier-free guidance** (Hyper-SD 4-step distillation bakes guidance in, so `guidance = 1.0` is the default and fastest mode). To steer generations away from unwanted styles, apply CFG in the client — encode a negative prompt and run `denoise` twice per step:

$$\epsilon = \epsilon_{\text{neg}} + w \cdot (\epsilon_{\text{pos}} - \epsilon_{\text{neg}})$$

A proven negative prompt for this checkpoint is `cartoon, cgi, render, illustration, painting, drawing`. Keep $w$ low ($\le 2.5$) — high CFG oversaturates the distilled output. See [`generate_gallery.py`](generate_gallery.py) for a working implementation.

---

## React Native Client Integration

```typescript
import React, { useState } from 'react';
import { View, Button, Text } from 'react-native';
import { useExecutorchModule } from 'react-native-executorch';
import { Canvas, Skia, Image as SkiaImage } from '@shopify/react-native-skia';
import { float32TensorToSkiaImage } from './integration_snippet';

export const EPICREALISM_CONFIG = {
  modelUri: 'https://huggingface.co/heymeowcat/epicrealism-512-executorch/resolve/main/xnnpack/epicrealism_512_xnnpack_fp32.pte',
  timesteps: [999, 759, 499, 259],
};

export function GenerationExample() {
  const { module, isLoaded } = useExecutorchModule(EPICREALISM_CONFIG.modelUri);
  const [skiaImg, setSkiaImg] = useState(null);

  const generate = async (tokenIds) => {
    // 1. Encode text prompt
    const embeds = await module.forward('encode', [tokenIds]);

    // 2. 4-step Denoising loop
    let latents = new Float32Array(1 * 4 * 64 * 64); // Gaussian noise
    for (const t of EPICREALISM_CONFIG.timesteps) {
      const noise = await module.forward('denoise', [latents, new BigInt64Array([BigInt(t)]), embeds]);
      latents = noise; // TCD step update
    }

    // 3. Decode with Full VAE
    const rgbTensor = await module.forward('decode', [latents]);
    setSkiaImg(float32TensorToSkiaImage(rgbTensor, 512, 512));
  };

  return (
    <View>
      {skiaImg && <Canvas style={{ width: 340, height: 340 }}><SkiaImage image={skiaImg} width={340} height={340} /></Canvas>}
      <Button title="Generate" onPress={() => generate(/* tokens */)} disabled={!isLoaded} />
    </View>
  );
}
```

See [integration_snippet.tsx](integration_snippet.tsx) for the complete production component with progress updates and input handling.

---

## Reproduction & Pipeline Execution

```bash
# 1. Create environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download source weights + tokenizer (~2.3 GB)
./download_epic.sh

# 3. Export ExecuTorch (.pte) model
python export_epicrealism_pte.py

# 4. Verify via ExecuTorch runtime and produce benchmark preview
python verify_epicrealism_pte.py

# 5. (Optional) Refresh the showcase gallery
python generate_gallery.py
```

---

## License

This model is released under the [CreativeML Open RAIL++-M License](LICENSE) (`license: openrail++`).

It is a derivative work of:
- [emilianJR/epiCRealism](https://huggingface.co/emilianJR/epiCRealism) (based on Stable Diffusion 1.5)
- [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) Hyper-SD15 4-step LoRA
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) VAE decoder

All upstream use-based restrictions (see **Annex A — Limitation of Use** in the [LICENSE](LICENSE)) continue to apply to this model and its outputs.
