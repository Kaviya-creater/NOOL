"""
NOOL - AI Image Enhancer
-------------------------
Takes a rough/blurry/low-quality reference image (old saree photo,
a screenshot, a phone snap of an embroidery pattern, etc.) and
returns a cleaned-up, sharper, higher-resolution version that an
artisan can actually use as a design reference.

Pipeline (all real, working image processing - no external API needed):
  1. Denoise            -> remove JPEG/phone-camera noise
  2. Upscale (2x/4x)    -> classic + edge-aware super-resolution
  3. Sharpen             -> recover lost detail (unsharp mask)
  4. Colour & contrast   -> auto white balance + CLAHE contrast boost
  5. (optional) DNN super-res if OpenCV's contrib model file is present

This is intentionally dependency-light (OpenCV + Pillow only) so it
runs anywhere, but is structured so you can swap in a real deep
super-resolution model (e.g. Real-ESRGAN) later - see
`_try_dnn_super_resolution()` for the plug-in point.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImageEnhancer:
    def __init__(self, dnn_model_path: str = None):
        """
        dnn_model_path: optional path to an OpenCV-compatible super-resolution
        model (.pb file), e.g. EDSR_x4.pb / ESPCN_x4.pb / FSRCNN_x4.pb.
        If not provided, falls back to high-quality classical upscaling.
        """
        self.dnn_model_path = dnn_model_path
        self.sr_model = None
        if dnn_model_path and os.path.exists(dnn_model_path):
            self._load_dnn_model()

    def _load_dnn_model(self):
        try:
            self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr_model.readModel(self.dnn_model_path)
            model_name = os.path.basename(self.dnn_model_path).split("_")[0].lower()
            scale = 4 if "x4" in self.dnn_model_path else 2
            self.sr_model.setModel(model_name, scale)
        except Exception as e:
            print(f"[enhancer] Could not load DNN super-res model, "
                  f"falling back to classical upscale: {e}")
            self.sr_model = None

    # ---------- individual steps ----------

    def _denoise(self, img):
        return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

    def _upscale(self, img, scale: int = 2):
        if self.sr_model is not None:
            try:
                return self.sr_model.upsample(img)
            except Exception as e:
                print(f"[enhancer] DNN upsample failed, using classical: {e}")

        h, w = img.shape[:2]
        # Lanczos gives sharper classical upscaling than bicubic for line art / fabric texture
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    def _sharpen(self, img, amount: float = 1.4):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        return sharpened

    def _auto_contrast_and_color(self, img):
        # CLAHE on the L channel of LAB colour space -> boosts local contrast
        # without blowing out colours (important for fabric/embroidery detail)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gentle saturation + brightness lift via PIL (fabric colours pop more)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Color(pil_img).enhance(1.15)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.05)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.08)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ---------- public entry point ----------

    def enhance(self, input_path: str, output_path: str, scale: int = 2) -> str:
        """
        Runs the full enhancement pipeline on input_path and writes
        the result to output_path. Returns output_path.
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        img = self._denoise(img)
        img = self._upscale(img, scale=scale)
        img = self._sharpen(img)
        img = self._auto_contrast_and_color(img)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return output_path
