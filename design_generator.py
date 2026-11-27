import io
import urllib.parse
import logging
import requests
from PIL import Image

logger = logging.getLogger(__name__)

class DesignGenerator:
    def __init__(self):
        # Base endpoint using FLUX without credit limits or tokens
        self.base_url = "https://image.pollinations.ai/prompt/"

    def generate(self, garment_type="", fabric="", colour="", embroidery="", style_notes="", custom_prompt=""):
        if custom_prompt and custom_prompt.strip():
            full_prompt = custom_prompt.strip()
        else:
            descriptors = [garment_type, fabric, colour, embroidery, style_notes]
            valid_parts = [d.strip() for d in descriptors if d and d.strip()]
            valid_parts.append("high quality textile weave, studio lighting, detailed fabric texture, photorealistic")
            full_prompt = ", ".join(valid_parts)

        logger.info(f"Generating image for prompt: {full_prompt}")

        # URL-encode prompt and configure parameters
        encoded_prompt = urllib.parse.quote(full_prompt)
        request_url = f"{self.base_url}{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true&seed=42"

        try:
            response = requests.get(request_url, timeout=90)
            if response.status_code != 200:
                raise RuntimeError(f"Generation API returned HTTP {response.status_code}")

            # Verify the response is a valid image
            img = Image.open(io.BytesIO(response.content))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            return {
                "success": True,
                "prompt": full_prompt,
                "source": "pollinations/flux",
                "image_bytes": buffer.getvalue()
            }
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise RuntimeError(f"Design generation failed: {str(e)}")
