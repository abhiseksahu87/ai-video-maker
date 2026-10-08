
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


# ============================================================
# APP SETUP
# ============================================================

APP_DIR = Path(__file__).resolve().parent

WORK_DIR = APP_DIR / "jobs"
WORK_DIR.mkdir(exist_ok=True)

JOBS = {}
JOBS_LOCK = Lock()

app = FastAPI(
    title="AI YouTube Video Maker"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class CompleteVideoRequest(BaseModel):

    topic: str = Field(
        min_length=3,
        max_length=500
    )

    language: str = "Hindi"

    duration_minutes: int = Field(
        default=10,
        ge=1,
        le=30
    )


# ============================================================
# JOB HELPERS
# ============================================================

def set_job(job_id, **values):

    with JOBS_LOCK:

        JOBS.setdefault(
            job_id,
            {}
        ).update(values)


def get_job(job_id):

    with JOBS_LOCK:

        return dict(
            JOBS.get(
                job_id,
                {}
            )
        )


# ============================================================
# GEMINI CLIENT
# ============================================================

def gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# SCRIPT GENERATION
# ============================================================

def make_script(
    client,
    topic,
    language,
    duration
):

    prompt = f"""
Create a professional long-form YouTube documentary script.

Topic: {topic}

Language: {language}

Target duration: {duration} minutes.

The documentary should feel cinematic,
professional and suitable for YouTube.

Use this exact structure:

TITLE:
A strong YouTube title.

HOOK:
A powerful opening.

INTRO:
A short introduction.

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

Continue with enough scenes to cover
the requested duration.

ENDING:
A strong conclusion and natural
YouTube call-to-action.

IMPORTANT FACTUAL REQUIREMENTS:

- Prioritize historically accurate
  and evidence-based information.
- Clearly distinguish established
  evidence from traditional,
  legendary, religious or disputed
  accounts.
- Never invent dates.
- Never invent quotations.
- Never invent battles.
- Never invent numbers.
- Never invent archaeological discoveries.
- Never invent historical events.
- Acknowledge uncertainty when historians
  disagree.
- Avoid exaggerated claims.
- Avoid anachronisms.

VISUAL REQUIREMENTS:

- Every scene must have a useful
  visual description.
- Historical scenes must use
  period-appropriate clothing,
  architecture, weapons and technology.
- Do not invent an exact appearance
  of a historical person when evidence
  is unavailable.
- Make visuals cinematic.
- Do not add logos.
- Do not add watermarks.
- Do not add modern objects.

SCRIPT REQUIREMENTS:

- Make the story flow naturally.
- Keep narration engaging.
- Avoid unnecessary repetition.
- Make scene durations approximately
  match the requested duration.
- Make the narration suitable for
  professional voiceover.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
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
        r"Scene\s+\d+\s*:\s*(.*?)"
        r"(?=\n\s*Scene\s+\d+\s*:|"
        r"\n\s*ENDING\s*:|\Z)",
        re.I | re.S
    )

    scenes = []

    blocks = pattern.findall(
        script
    )

    for block in blocks:

        narration_match = re.search(
            r"Narration\s*:\s*(.*?)"
            r"(?=\n\s*Visual\s*:)",
            block,
            re.I | re.S
        )

        visual_match = re.search(
            r"Visual\s*:\s*(.*?)"
            r"(?=\n\s*Duration\s*:|\Z)",
            block,
            re.I | re.S
        )

        duration_match = re.search(
            r"Duration\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)",
            block,
            re.I
        )

        if not narration_match:
            continue

        narration = (
            narration_match
            .group(1)
            .strip()
        )

        if visual_match:

            visual = (
                visual_match
                .group(1)
                .strip()
            )

        else:

            visual = (
                "Cinematic documentary "
                "scene related to the narration."
            )

        if duration_match:

            duration = float(
                duration_match.group(1)
            )

        else:

            duration = 60.0

        if narration:

            scenes.append(
                {
                    "narration": narration,
                    "visual": visual,
                    "duration": duration
                }
            )

    return scenes


# ============================================================
# CREATE COMPLETE NARRATION
# ============================================================

def make_narration(
    scenes,
    script
):

    if scenes:

        return "\n\n".join(
            scene["narration"]
            for scene in scenes
        )

    return script


# ============================================================
# VOICEOVER GENERATION
# ============================================================

def generate_voice(
    client,
    narration,
    language,
    output_path
):

    voice_prompt = f"""
Read this documentary narration exactly
as written.

Language: {language}

Voice style:

- Natural
- Clear
- Warm
- Authoritative
- Cinematic
- Moderate speaking pace
- Appropriate dramatic emphasis
- No background music
- Do not add words

NARRATION:

{narration}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash-tts",

        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": voice_prompt
                    }
                ]
            }
        ],

        config={
            "response_modalities": [
                "AUDIO"
            ],

            "speech_config": {
                "voice_config": {
                    "voice": "Kore"
                }
            }
        }
    )

    audio_data = None

    if response.candidates:

        candidate = (
            response.candidates[0]
        )

        if (
            candidate.content
            and candidate.content.parts
        ):

            for part in (
                candidate.content.parts
            ):

                if getattr(
                    part,
                    "inline_data",
                    None
                ):

                    audio_data = (
                        part.inline_data.data
                    )

                    break

    if not audio_data:

        raise RuntimeError(
            "Gemini returned no audio."
        )

    if isinstance(
        audio_data,
        str
    ):

        audio_bytes =
