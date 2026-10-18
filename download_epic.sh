#!/bin/bash
set -e
mkdir -p models/epicrealism models/hyper_sd models/swmansion_tokenizer

H="https://huggingface.co"
d(){
    mkdir -p "$(dirname "$1")"
    echo ">>> Downloading $1 from $2"
    aria2c -x16 -s16 -k4M --file-allocation=none -c -d "$(dirname "$1")" -o "$(basename "$1")" "$2"
    echo "OK: $1"
}

# 1. epiCRealism newEra pruned safetensors (2.03 GB)
d "models/epicrealism/epicrealism_newEra.safetensors" "$H/Robocrack/epicrealism/resolve/main/epicrealism_newEra.safetensors"

# 2. Hyper-SD 4-steps LoRA (256 MB)
d "models/hyper_sd/Hyper-SD15-4steps-lora.safetensors" "$H/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-4steps-lora.safetensors"

# 3. CLIP ViT-L/14 tokenizer (3.5 MB) — from Software Mansion's RN ExecuTorch SDXS export
d "models/swmansion_tokenizer/tokenizer.json" "$H/software-mansion/react-native-executorch-sdxs-512-dreamshaper/resolve/main/tokenizer.json"
d "models/swmansion_tokenizer/tokenizer_config.json" "$H/software-mansion/react-native-executorch-sdxs-512-dreamshaper/resolve/main/tokenizer_config.json"

echo "ALL_DOWNLOADS_COMPLETED"
