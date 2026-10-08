
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from google import genai

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI YouTube Video Maker")


class ScriptRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    language: str = "English"
    duration_minutes: int = Field(default=10, ge=1, le=60)


@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/script")
def generate_script(req: ScriptRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )

    prompt = f"""
Create a professional YouTube documentary script.

Topic: {req.topic}
Language: {req.language}
Target duration: {req.duration_minutes} minutes.

Create a compelling long-form video suitable for YouTube.

Return the result in this exact structure:

TITLE:
A strong YouTube title

HOOK:
An engaging opening narration.

INTRO:
A short introduction.

SCENES:
Scene 1:
Narration:
Visual:
Duration:

Scene 2:
Narration:
Visual:
Duration:

Continue until the complete video is covered.

ENDING:
A strong conclusion and call to action.

Requirements:
- Make the narration historically/informationally responsible.
- Use cinematic visual descriptions.
- Keep each scene suitable for later image/video generation.
- Make the story flow naturally from beginning to end.
- Do not mention these instructions.
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return {
            "success": True,
            "script": response.text
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
