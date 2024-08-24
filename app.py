import io
import modal

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
    )

app = modal.App('flux1')

with image.imports():
    import torch
    from diffusers import FluxPipeline
    from fastapi import Response
        
@app.cls(gpu=modal.gpu.A100(), container_idle_timeout=15, image=image, timeout=120)
class Model:
    @modal.build()
    def build(self):
        from huggingface_hub import snapshot_download

        snapshot_download("black-forest-labs/FLUX.1-schnell")

    @modal.enter()
    def enter(self):
        print("Loading model...")
        self.pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to('cuda')
        print("Model loaded!")

    def inference(self, prompt: str, width: int = 1440, height: int = 1440):
        print("Generating image...")
        image = self.pipeline(
            prompt, 
            output_type='pil', 
            width=width, 
            height=height, 
            num_inference_steps=4,
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
    
    @modal.web_endpoint(docs=True)
    def web_inference(self, prompt: str, width: int = 1440, height: int = 1440):
        image = self.inference(prompt, width, height)
        return Response(content=image, media_type="image/png")
    
@app.local_entrypoint()
def main(prompt: str = "A beautiful sunset over the mountains"):
    image_bytes = Model()._inference.remote(prompt)

    with open("output.png", "wb") as f:
        f.write(image_bytes)
