"""
Runpod Serverless Handler for Omost
This handler processes image generation requests using the Omost pipeline.
"""

import os
import sys
import base64
import io
import traceback
from typing import Dict, Any

# Set up paths
sys.path.append('/app')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')

import torch
import numpy as np
from PIL import Image
import runpod

# Import Omost modules
import lib_omost.memory_management as memory_management
import lib_omost.canvas as omost_canvas
from lib_omost.pipeline import StableDiffusionXLOmostPipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

# Phi3 Hijack
Phi3PreTrainedModel._supports_sdpa = True

# Global variables for models
llm_model = None
llm_tokenizer = None
pipeline = None
text_encoder = None
text_encoder_2 = None
vae = None
unet = None


def load_models():
    """Load all required models on startup"""
    global llm_model, llm_tokenizer, pipeline
    global text_encoder, text_encoder_2, vae, unet
    
    print("Loading models...")
    
    # Get configuration from environment variables
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    llm_name = os.environ.get('LLM_MODEL', 'lllyasviel/omost-llama-3-8b-4bits')
    sdxl_name = os.environ.get('SDXL_MODEL', 'SG161222/RealVisXL_V4.0')
    
    # Load SDXL components
    print(f"Loading SDXL model: {sdxl_name}")
    tokenizer = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16"
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(
        sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16"
    )
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )
    
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())
    
    pipeline = StableDiffusionXLOmostPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=None,
    )
    
    memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])
    
    # Load LLM
    print(f"Loading LLM model: {llm_name}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        device_map="auto"
    )
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, token=HF_TOKEN)
    
    memory_management.unload_all_models(llm_model)
    
    print("All models loaded successfully!")


@torch.inference_mode()
def generate_canvas_from_prompt(prompt: str, seed: int = 12345, 
                                temperature: float = 0.6, top_p: float = 0.9,
                                max_new_tokens: int = 4096) -> Dict[str, Any]:
    """Generate canvas from text prompt using LLM"""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    
    conversation = [
        {"role": "system", "content": omost_canvas.system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    memory_management.load_models_to_gpu(llm_model)
    
    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    ).to(llm_model.device)
    
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    if temperature == 0:
        generate_kwargs['do_sample'] = False
    
    output = llm_model.generate(**generate_kwargs)
    response = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    if "assistant" in response.lower():
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    # Parse canvas from response
    canvas = omost_canvas.Canvas.from_bot_response(response)
    canvas_outputs = canvas.process()
    
    return canvas_outputs


@torch.inference_mode()
def pytorch2numpy(imgs):
    """Convert PyTorch tensors to numpy arrays"""
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def generate_image(canvas_outputs: Dict[str, Any], seed: int = 12345,
                  image_width: int = 896, image_height: int = 1152,
                  num_samples: int = 1, steps: int = 25, cfg: float = 5.0,
                  negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality") -> list:
    """Generate images from canvas outputs"""
    
    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64
    
    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)
    
    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
    
    positive_cond, positive_pooler, negative_cond, negative_pooler = \
        pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)
    
    initial_latent = torch.zeros(
        size=(num_samples, 4, image_height // 8, image_width // 8),
        dtype=torch.float32
    )
    
    memory_management.load_models_to_gpu([unet])
    
    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)
    
    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images
    
    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    
    return pixels


def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Encode numpy array image to base64 string"""
    img = Image.fromarray(image_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod handler function
    
    Expected input format:
    {
        "input": {
            "prompt": "your text prompt",
            "seed": 12345,  # optional
            "num_samples": 1,  # optional
            "steps": 25,  # optional
            "cfg": 5.0,  # optional
            "image_width": 896,  # optional
            "image_height": 1152,  # optional
            "temperature": 0.6,  # optional, for LLM
            "top_p": 0.9,  # optional, for LLM
            "max_new_tokens": 4096,  # optional, for LLM
            "negative_prompt": "lowres, bad anatomy..."  # optional
        }
    }
    """
    try:
        # Extract input parameters
        job_input = event.get('input', {})
        
        prompt = job_input.get('prompt')
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Get parameters with defaults
        seed = job_input.get('seed', int(os.environ.get('SEED', 12345)))
        num_samples = job_input.get('num_samples', 1)
        steps = job_input.get('steps', int(os.environ.get('DEFAULT_STEPS', 25)))
        cfg = job_input.get('cfg', float(os.environ.get('DEFAULT_CFG', 5.0)))
        image_width = job_input.get('image_width', int(os.environ.get('DEFAULT_IMAGE_WIDTH', 896)))
        image_height = job_input.get('image_height', int(os.environ.get('DEFAULT_IMAGE_HEIGHT', 1152)))
        temperature = job_input.get('temperature', float(os.environ.get('TEMPERATURE', 0.6)))
        top_p = job_input.get('top_p', float(os.environ.get('TOP_P', 0.9)))
        max_new_tokens = job_input.get('max_new_tokens', int(os.environ.get('MAX_NEW_TOKENS', 4096)))
        negative_prompt = job_input.get('negative_prompt', 
                                       os.environ.get('NEGATIVE_PROMPT', 
                                                     'lowres, bad anatomy, bad hands, cropped, worst quality'))
        
        print(f"Processing request: prompt='{prompt}', seed={seed}")
        
        # Step 1: Generate canvas from prompt
        print("Generating canvas from prompt...")
        canvas_outputs = generate_canvas_from_prompt(
            prompt=prompt,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        
        # Step 2: Generate images from canvas
        print("Generating images...")
        images = generate_image(
            canvas_outputs=canvas_outputs,
            seed=seed,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            steps=steps,
            cfg=cfg,
            negative_prompt=negative_prompt
        )
        
        # Step 3: Encode images to base64
        print("Encoding images...")
        encoded_images = [encode_image_to_base64(img) for img in images]
        
        return {
            "images": encoded_images,
            "seed": seed,
            "prompt": prompt,
            "num_images": len(encoded_images)
        }
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}


if __name__ == "__main__":
    print("Starting Runpod Serverless handler for Omost...")
    
    # Load models on startup
    load_models()
    
    print("Starting Runpod serverless function...")
    runpod.serverless.start({"handler": handler})

