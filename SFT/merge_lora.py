from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
import torch


def load_flux2_klein_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Flux2KleinPipeline.from_pretrained(
        "/mnt/shared-storage-user/mllm/zhangbeichen/models/models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/17c3b160520b7dd44665dbf0b9ed9dd30c15cd06",
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights("/mnt/shared-storage-user/mllm/zhangbeichen/results_klein/0224/epoch-0.safetensors", adapter_name="default")
    pipe.to(device)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.save_pretrained("/mnt/shared-storage-user/mllm/zhangbeichen/share_models/ETCHR-SFT", safe_serialization=True)

load_flux2_klein_pipeline()