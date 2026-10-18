"""Export epiCRealism + Hyper-SD 4-Step + Full VAE Decoder to ExecuTorch (.pte)
in full unquantized FP32 for 100% mathematical parity with the eager PyTorch pipeline.
Zero quantization loss, zero time-embedding distortion.

Methods:
  encode:  int64[1, 77]                              -> float32[1, 77, 768]
  denoise: float32[1, 4, 64, 64], int64[1], float32[1, 77, 768] -> float32[1, 4, 64, 64]
  decode:  float32[1, 4, 64, 64]                   -> float32[1, 3, 512, 512] in [0, 1]
"""
import gc
import os
import sys
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import transformers
from transformers import CLIPImageProcessor

setattr(transformers, "CLIPFeatureExtractor", CLIPImageProcessor)
sys.modules["transformers"].CLIPFeatureExtractor = CLIPImageProcessor

torch.set_grad_enabled(False)

BASE_CHECKPOINT = "models/epicrealism/epicrealism_newEra.safetensors"
LORA_4STEP = "models/hyper_sd/Hyper-SD15-4steps-lora.safetensors"
OUT_PTE = "out/epicrealism_512_xnnpack_fp32.pte"
FP32 = torch.float32

# ------------------------------------------------------------ 1) Load + Fuse LoRA
print("[1/5] Loading epiCRealism + fusing Hyper-SD 4-Step LoRA (FP32) ...", flush=True)
pipe = StableDiffusionPipeline.from_single_file(
    os.path.abspath(BASE_CHECKPOINT),
    torch_dtype=FP32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.load_lora_weights(os.path.abspath(LORA_4STEP))
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()

text_encoder = pipe.text_encoder.eval()
unet = pipe.unet.eval()
post_quant_conv = pipe.vae.post_quant_conv.eval()
decoder = pipe.vae.decoder.eval()
scaling_factor = float(pipe.vae.config.scaling_factor)
del pipe.vae, pipe
gc.collect()

# ------------------------------------------------------------ 2) Module Definitions
class EncodeModule(nn.Module):
    def __init__(self, te):
        super().__init__()
        self.net = te

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(input_ids=tokens).last_hidden_state


class DenoiseModule(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.net = u

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        out = self.net(latents, timestep, embeds)
        return out.sample if hasattr(out, "sample") else out


class DecodeModule(nn.Module):
    """Full VAE Decode: post_quant_conv -> decoder -> [0, 1] RGB."""
    def __init__(self, post_conv, dec, scale):
        super().__init__()
        self.post_quant_conv = post_conv
        self.decoder = dec
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(latents / self.scale)
        img = self.decoder(z)
        return (img / 2.0 + 0.5).clamp(0.0, 1.0)


encode_m = EncodeModule(text_encoder).eval()
denoise_m = DenoiseModule(unet).eval()
decode_m = DecodeModule(post_quant_conv, decoder, scaling_factor).eval()

# ------------------------------------------------------------ 3) Export Each Method
print("[2/5] Tracing methods with torch.export ...", flush=True)
tokens_ex = torch.ones((1, 77), dtype=torch.int64)
lat_ex = torch.randn(1, 4, 64, 64)
t_ex = torch.tensor([999], dtype=torch.int64)
emb_ex = encode_m(tokens_ex)

ep_encode = torch.export.export(encode_m, (tokens_ex,))
ep_denoise = torch.export.export(denoise_m, (lat_ex, t_ex, emb_ex))
ep_decode = torch.export.export(decode_m, (lat_ex,))
print("    Exported encode, denoise, decode successfully", flush=True)

del text_encoder, unet, post_quant_conv, decoder
gc.collect()

# ------------------------------------------------------------ 4) Lower to XNNPACK
print("[3/5] Lowering to ExecuTorch with XNNPACK FP32 partitioner ...", flush=True)
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackFloatingPointPartitioner

edge = to_edge_transform_and_lower(
    {"encode": ep_encode, "denoise": ep_denoise, "decode": ep_decode},
    partitioner={
        "encode": [XnnpackFloatingPointPartitioner()],
        "denoise": [XnnpackFloatingPointPartitioner()],
        "decode": [XnnpackFloatingPointPartitioner()],
    },
    compile_config=None,
)

print("[4/5] Serializing to ExecuTorch binary ...", flush=True)
et = edge.to_executorch()
os.makedirs(os.path.dirname(OUT_PTE), exist_ok=True)
with open(OUT_PTE, "wb") as f:
    f.write(et.buffer)

size_mb = os.path.getsize(OUT_PTE) / 1024 / 1024
print(f"[5/5] Successfully wrote {OUT_PTE} ({size_mb:.1f} MB)", flush=True)
print("EXPORT_EPICREALISM_FP32_OK", flush=True)
