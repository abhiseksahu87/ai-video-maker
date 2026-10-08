import os, time, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

APP_DIR = Path(__file__).resolve().parent
GENERATED_DIR = APP_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
app = FastAPI(title="AI Video Maker")

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)
    aspect_ratio: str = "9:16"
    resolution: str = "720p"

@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/generate")
def generate_video(req: GenerateRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    if req.aspect_ratio not in {"9:16", "16:9"}:
        raise HTTPException(status_code=400, detail="Invalid aspect ratio.")
    if req.resolution not in {"720p", "1080p"}:
        raise HTTPException(status_code=400, detail="Invalid resolution.")
    try:
        client = genai.Client(api_key=api_key)
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=req.prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio,
                resolution=req.resolution,
            ),
        )
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)
        if not operation.response or not operation.response.generated_videos:
            raise RuntimeError("Veo returned no generated video.")
        video = operation.response.generated_videos[0]
        filename = f"{uuid.uuid4().hex}.mp4"
        output_path = GENERATED_DIR / filename
        client.files.download(file=video.video, destination=str(output_path))
        return {"success": True, "video_url": f"/api/video/{filename}", "filename": filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/video/{filename}")
def get_video(filename: str):
    safe_name = Path(filename).name
    path = GENERATED_DIR / safe_name
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)
