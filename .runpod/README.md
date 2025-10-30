# Omost on Runpod Serverless

This directory contains the configuration files and handler for running Omost on Runpod Serverless.

## Quick Deploy

[![Runpod](https://api.runpod.io/badge/DerSpirer/Omost)](https://console.runpod.io/hub/DerSpirer/Omost)

Click the badge above to instantly deploy Omost to Runpod!

## What's Included

- **hub.json**: Configuration for the Runpod Hub listing
- **tests.json**: Test cases to validate the deployment
- **Dockerfile**: Container build instructions
- **handler.py**: Serverless handler that processes image generation requests
- **README.md**: This file

## API Usage

Once deployed, you can use the Runpod API to generate images:

### Request Format

```json
{
  "input": {
    "prompt": "a serene landscape with mountains and a lake at sunset",
    "seed": 12345,
    "num_samples": 1,
    "steps": 25,
    "cfg": 5.0,
    "image_width": 896,
    "image_height": 1152,
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": 4096,
    "negative_prompt": "lowres, bad anatomy, bad hands, cropped, worst quality"
  }
}
```

### Parameters

#### Required
- **prompt** (string): Text description of the image you want to generate

#### Optional
- **seed** (int): Random seed for reproducibility (default: 12345)
- **num_samples** (int): Number of images to generate (default: 1)
- **steps** (int): Number of diffusion steps (default: 25)
- **cfg** (float): Classifier-free guidance scale (default: 5.0)
- **image_width** (int): Image width in pixels, must be multiple of 64 (default: 896)
- **image_height** (int): Image height in pixels, must be multiple of 64 (default: 1152)
- **temperature** (float): LLM temperature for creativity (default: 0.6)
- **top_p** (float): LLM top-p sampling (default: 0.9)
- **max_new_tokens** (int): Max tokens for LLM to generate (default: 4096)
- **negative_prompt** (string): What to avoid in the image

### Response Format

```json
{
  "images": ["base64_encoded_image_1", "base64_encoded_image_2"],
  "seed": 12345,
  "prompt": "your prompt here",
  "num_images": 2
}
```

The images are returned as base64-encoded PNG files.

## Python Example

```python
import runpod
import base64
from PIL import Image
from io import BytesIO

# Configure your API key
runpod.api_key = "your_runpod_api_key"

# Create endpoint
endpoint = runpod.Endpoint("ENDPOINT_ID")

# Run inference
result = endpoint.run_sync({
    "input": {
        "prompt": "a fierce battle between warriors and a dragon",
        "seed": 12345,
        "num_samples": 1,
        "steps": 25,
        "cfg": 5.0,
        "image_width": 1024,
        "image_height": 1024
    }
})

# Decode and save images
if "images" in result:
    for i, img_base64 in enumerate(result["images"]):
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        img.save(f"output_{i}.png")
```

## Configuration Presets

The deployment includes three presets optimized for different use cases:

### 1. Quality (Llama3-8B) - Recommended
- Best overall quality and prompt following
- Uses 4-bit quantized Llama3-8B model
- Safe for production use (filtered training data)

### 2. Uncensored (Dolphin Llama3)
- No content filtering
- Uses community-trained Dolphin variant
- **Note**: Apply your own safety measures if exposing publicly

### 3. Compact (Phi3-Mini)
- Smaller model for faster inference
- Lower GPU memory requirements
- Slightly lower quality than Llama3

## Model Options

### LLM Models
- `lllyasviel/omost-llama-3-8b-4bits` (Recommended)
- `lllyasviel/omost-llama-3-8b`
- `lllyasviel/omost-dolphin-2.9-llama3-8b-4bits`
- `lllyasviel/omost-dolphin-2.9-llama3-8b`
- `lllyasviel/omost-phi-3-mini-128k-8bits`
- `lllyasviel/omost-phi-3-mini-128k`

### SDXL Models
- `SG161222/RealVisXL_V4.0` (Default)
- `stabilityai/stable-diffusion-xl-base-1.0`
- Any other SDXL-compatible model from Hugging Face

## GPU Requirements

- **Minimum**: 16GB VRAM (RTX 4090, A4000, etc.)
- **Recommended**: 24GB+ VRAM for better performance
- **Container Disk**: 40GB (for model storage)

## Environment Variables

All configuration can be set via environment variables:

- `HF_TOKEN`: Hugging Face token (for gated models)
- `LLM_MODEL`: Which Omost LLM to use
- `SDXL_MODEL`: Which SDXL model to use
- `DEFAULT_STEPS`, `DEFAULT_CFG`: Default generation parameters
- `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`: LLM parameters

## Testing Locally

Before deploying, you can test the handler locally:

```bash
# Build the Docker image
docker build -t omost-runpod -f .runpod/Dockerfile .

# Run with environment variables
docker run --gpus all \
  -e LLM_MODEL=lllyasviel/omost-llama-3-8b-4bits \
  -e SDXL_MODEL=SG161222/RealVisXL_V4.0 \
  omost-runpod
```

## Publishing to Runpod Hub

Follow these steps to publish:

1. **Navigate** to the Hub page in Runpod console
2. **Click** "Add your repo" → "Get Started"
3. **Enter** your GitHub repo URL
4. **Create** a new GitHub release (Hub indexes releases, not commits)
5. **Wait** for building/testing (usually within an hour)
6. **Await** manual review by Runpod team

## Updating Your Deployment

To update your Hub listing, simply create a new GitHub release. The Hub will automatically rebuild and test within an hour.

## Support

For issues specific to this Runpod deployment, please open an issue on the GitHub repository.

For general Omost questions, refer to the [main README](../README.md).

