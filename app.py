import os
import re
import base64
import uuid
import wave
import subprocess
from pathlib import Path
from threading import Thread, Lock

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from google import genai
import imageio_ffmpeg


APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "jobs"
WORK_DIR.mkdir(exist_ok=True)

JOBS = {}
JOBS_LOCK = Lock()

app = FastAPI(title="AI YouTube Video Maker")


class CompleteVideoRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    language: str = "Hindi"
    duration_minutes: int = Field(default=10, ge=1, le=30)


def set_job(job_id, **values):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(values)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


def gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# SCRIPT GENERATION
# ============================================================

def make_script(client, topic, language, duration):

    prompt = f"""
Create a professional long-form YouTube documentary script.

Topic: {topic}
Language: {language}
Target duration: {duration} minutes.

Return exactly this structure:

TITLE:
...

HOOK:
...

INTRO:
...

SCENES:

Scene 1:
Narration:
...
Visual:
...
Duration:
...

Scene 2:
Narration:
...
Visual:
...
Duration:
...

Continue with enough scenes for the target duration.

ENDING:
...

FACTUAL REQUIREMENTS:

- Prioritize evidence-based information.
- Clearly distinguish established evidence from traditional,
  legendary, religious, or disputed accounts.
- Never invent dates, quotations, battles, numbers, places,
  people, discoveries, or events.
- Acknowledge important uncertainty.
- Avoid exaggerated claims unless well supported.
- Avoid anachronisms.

VISUAL REQUIREMENTS:

- Every scene needs a useful visual description.
- Historical visuals must use period-appropriate clothing,
  architecture, landscapes and technology.
- Do not invent a person's exact appearance when reliable
  evidence is absent.
- Make visuals suitable for a cinematic documentary.
- Do not put text, logos, captions or watermarks inside visuals.

Keep narration natural and suitable for voiceover.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty script."
        )

    return response.text


# ============================================================
# EXTRACT SCENES
# ============================================================

def extract_scenes(script):

    pattern = re.compile(
        r"Scene\s+\d+\s*:\s*(.*?)(?=\n\s*Scene\s+\d+\s*:|\n\s*ENDING\s*:|\Z)",
        re.I | re.S,
    )

    scenes = []

    for block in pattern.findall(script):

        narration_match = re.search(
            r"Narration\s*:\s*(.*?)(?=\n\s*Visual\s*:)",
            block,
            re.I | re.S,
        )

        visual_match = re.search(
            r"Visual\s*:\s*(.*?)(?=\n\s*Duration\s*:|\Z)",
            block,
            re.I | re.S,
        )

        duration_match = re.search(
            r"Duration\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            block,
            re.I,
        )

        if not narration_match:
            continue

        narration = narration_match.group(1).strip()

        visual = (
            visual_match.group(1).strip()
            if visual_match
            else "Cinematic documentary scene related to the narration."
        )

        duration = (
            float(duration_match.group(1))
            if duration_match
            else 60.0
        )

        if narration:
            scenes.append(
                {
                    "narration": narration,
                    "visual": visual,
                    "duration": duration,
                }
            )

    return scenes


def make_narration(scenes, script):

    if scenes:
        return "\n\n".join(
            scene["narration"]
            for scene in scenes
        )

    return script


# ============================================================
# VOICEOVER
# ============================================================

def generate_voice(
    client,
    narration,
    language,
    output_path,
):

    voice_prompt = f"""
Read this documentary narration exactly as written.

Language: {language}

Style:

- Natural
- Clear
- Warm
- Authoritative
- Cinematic
- Moderate pace
- Dramatic emphasis where appropriate
- No background music
- Do not add words

NARRATION:

{narration}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash-tts",
        contents=[
            {
                "role": "
