
import os
import sys
import threading
from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="OpenThai 2.1 OCR API",
    version="1.0.0"
)

MODEL_ID = "iapp/OpenThai2.1-OCR-Text-20261118"
MODEL_REVISION = "v2.1.4"
MAX_FILE_SIZE = 5 * 1024 * 1024

ocr_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_ocr():
    model_dir = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "*.onnx",
            "*.json",
            "*.txt",
            "runtime/*.py"
        ]
    )

    sys.path.insert(0, model_dir)

    from runtime.line_engine import LineOCR

    return LineOCR(model_dir, threads=1)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "OpenThai 2.1 OCR API",
        "model": MODEL_ID,
        "version": MODEL_REVISION
    }


@app.post("/ocr")
def recognize(
    file: UploadFile = File(...),
    x_api_key: str = Header(default="")
):
    expected_key = os.getenv("OCR_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="OCR_API_KEY is not configured"
        )

    import secrets

    if not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WebP are supported"
        )

    data = file.file.read(MAX_FILE_SIZE + 1)

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 5 MB"
        )

    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    try:
        with ocr_lock:
            ocr = get_ocr()
            result = ocr.read(data)

        return {
            "success": True,
            "text": result.get("text", ""),
            "lines": result.get("lines", []),
            "rotation": result.get("rotation", 0),
            "engine": "OpenThai 2.1 OCR Text"
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="OCR processing failed"
        )
