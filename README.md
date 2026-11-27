# NOOL — AI Design Studio & Image Enhancer

A prototype of NOOL's two AI-powered features:

- **AI Design Studio** — turn a text description (or garment specs: type,
  fabric, colour, embroidery) into a design image, ready to send to an
  artisan.
- **AI Image Enhancer** — clean up a rough/blurry/low-quality reference
  photo (an old saree photo, a screenshot, a phone snap) into a sharp,
  higher-resolution image the artisan can actually work from.

Built to slot into NOOL's "Core intelligence" layer described in the
SIH26090 pitch — this repo is the standalone module, so it can be
demoed on its own or wired into the full app.

## Features

| Feature | How it works |
|---|---|
| Text → Design | Calls a GenAI text-to-image API (Stability AI by default). Falls back to a labelled offline mock if no API key is set, so the demo always works. |
| Photo → Enhanced | Real image processing pipeline: denoise → upscale (Lanczos, or a plugged-in deep super-resolution model) → sharpen (unsharp mask) → CLAHE contrast + colour boost. No API key needed — runs fully offline. |

## Project structure

```
nool-ai-studio/
├── backend/
│   ├── app.py                     # Flask app + API routes
│   ├── modules/
│   │   ├── design_generator.py    # AI Design Studio logic
│   │   └── enhancer.py            # AI Image Enhancer logic
│   ├── static/
│   │   └── index.html             # Demo UI (two tabs)
│   ├── uploads/                   # user-uploaded images (gitignored)
│   ├── outputs/                   # generated/enhanced images (gitignored)
│   └── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional — enable real AI design generation (otherwise runs in mock mode):

```bash
export STABILITY_API_KEY="your-key-here"   # https://platform.stability.ai
```

Optional — enable deep super-resolution for the enhancer (otherwise uses
high-quality classical upscaling, which already works well):

```bash
# Download an OpenCV-compatible model, e.g. EDSR_x4.pb / FSRCNN_x4.pb
export SR_MODEL_PATH="/path/to/EDSR_x4.pb"
```

Run it:

```bash
python app.py
# open http://localhost:5000
```

## API

### `POST /api/generate` — AI Design Studio

Form fields: `garment_type`, `fabric`, `colour`, `embroidery`, `custom_prompt`

```bash
curl -X POST http://localhost:5000/api/generate \
  -d "garment_type=saree" -d "fabric=silk" -d "colour=maroon" \
  -d "embroidery=peacock motif gold border"
```

Response:
```json
{ "prompt": "...", "source": "stability", "image_url": "/api/file/design_xxx.png" }
```

### `POST /api/enhance` — AI Image Enhancer

Multipart form: `image` (file), `scale` (`2` or `4`)

```bash
curl -X POST http://localhost:5000/api/enhance \
  -F "image=@/path/to/rough_photo.jpg" -F "scale=2"
```

Response:
```json
{ "image_url": "/api/file/enhanced_xxx.png" }
```

## Pushing to GitHub

```bash
cd nool-ai-studio
git init
git add .
git commit -m "Add AI Design Studio + Image Enhancer prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/nool-ai-studio.git
git push -u origin main
```

## Notes / next steps

- Swap `design_generator.py`'s `_call_stability_api` for any other
  text-to-image provider (OpenAI Images, Replicate, a fine-tuned
  in-house model) without touching the rest of the app.
- `enhancer.py` is structured so a real deep super-resolution model
  (e.g. Real-ESRGAN) can be dropped in via `SR_MODEL_PATH` without
  changing the API contract.
- For production: add file-type/virus scanning on uploads, rate
  limiting on the generate endpoint (API cost control), and move
  `uploads/`/`outputs/` to cloud storage (S3/Firebase) instead of local disk.
