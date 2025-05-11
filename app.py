import io
import modal

def download_weights():
    import os
    from huggingface_hub import snapshot_download

    hf_token = os.getenv("HF_TOKEN")
    snapshot_download("black-forest-labs/FLUX.1-schnell", token=hf_token)

image = modal.Image.debian_slim(python_version="3.10").apt_install(
        "libglib2.0-0", 
        "libsm6", 
        "libxrender1", 
        "libxext6", 
        "ffmpeg", 
        "libgl1",
        "git"
    ).pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "invisible_watermark",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
    ).env({
        "HF_HUB_CACHE": "/cache",
    }).run_function(
        download_weights,
        secrets=[modal.Secret.from_name("flux.1-secret")],
    )

app = modal.App('flux1')

with image.imports():
    import os
    import torch
    from diffusers import FluxPipeline
    from fastapi import Response, Header
        
@app.cls(
    gpu='A100-40GB', 
    scaledown_window=15, 
    image=image, 
    timeout=120, 
    secrets=[modal.Secret.from_name("flux.1-secret")],
)
class Model:
    @modal.enter()
    def enter(self):
        print("Loading model...")
        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        print("Model loaded!")

    def inference(self, prompt: str, width: int = 1440, height: int = 1440):
        print("Generating image...")
        image = self.pipeline(
            prompt, 
            output_type='pil', 
            width=width, 
            height=height, 
            num_inference_steps=8,
            generator=torch.Generator("cpu").manual_seed(
                torch.randint(0, 1000000, (1,)).item()
            )
        ).images[0]

        print("Image generated!")

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")

        return byte_stream.getvalue()
    
    @modal.method()
    def _inference(self, prompt: str, width: int = 1440, height: int = 1440):
        return self.inference(prompt, width, height)
    
    @modal.fastapi_endpoint(docs=True)
    def web_inference(self, prompt: str, width: int = 1440, height: int = 1440, x_api_key: str = Header(None)):
        api_key = os.getenv("API_KEY")
        if x_api_key != api_key:
            return Response(content="Unauthorized", status_code=401)

        image = self.inference(prompt, width, height)
        return Response(content=image, media_type="image/png")
    
@app.local_entrypoint()
def main(prompt: str = "A beautiful sunset over the mountains"):
    image_bytes = Model()._inference.remote(prompt)

    with open("output.png", "wb") as f:
        f.write(image_bytes)
