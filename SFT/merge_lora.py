from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
import torch


def load_flux2_klein_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Flux2KleinPipeline.from_pretrained(
        "YOUR_FLUX2_KLEIN_MODEL_PATH",
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights("YOUR_LORA_PATH", adapter_name="default")
    pipe.to(device)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.save_pretrained("SFT_SAVE_PATH", safe_serialization=True)

load_flux2_klein_pipeline()