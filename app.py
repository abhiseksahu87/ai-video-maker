import os
import base64
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from google import genai


APP_DIR = Path(__file__).resolve().parent
AUDIO_DIR = APP_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI YouTube Video Maker")


class ScriptRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    language: str = "English"
    duration_minutes: int = Field(default=10, ge=1, le=60)


class VoiceRequest(BaseModel):
    text: str = Field(min_length=3, max_length=12000)
    language: str = "Hindi"


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
Create a professional long-form YouTube documentary script.

Topic: {req.topic}
Language: {req.language}
Target duration: {req.duration_minutes} minutes.

The video should feel like a professional cinematic documentary
suitable for a YouTube history, civilization, science, technology,
or educational channel.

Return the result in this exact structure:

TITLE:
A strong and engaging YouTube title.

HOOK:
A powerful opening narration that immediately creates curiosity.

INTRO:
A short introduction explaining what the viewer will discover.

SCENES:

Scene 1:
Narration:
Visual:
Duration:

Scene 2:
Narration:
Visual:
Duration:

Continue with enough scenes to cover the complete target duration.

ENDING:
A strong conclusion and natural YouTube call-to-action.

IMPORTANT HISTORICAL AND FACTUAL REQUIREMENTS:

- Prioritize historically accurate and evidence-based information.
- Clearly distinguish established historical evidence from traditional,
  legendary, religious, or disputed accounts.
- Never invent dates, quotations, battles, numbers, places, people,
  archaeological discoveries, or historical events.
- Do not present legends or later traditions as proven facts.
- When historians disagree about an important issue, briefly acknowledge
  the uncertainty.
- Avoid exaggerated claims unless they are well supported.
- Be especially careful with ancient history where primary evidence
  may be limited.
- If a claim comes mainly from a later tradition, explicitly say so.
- Keep the narration engaging without sacrificing factual accuracy.

VISUAL REQUIREMENTS:

- Give cinematic visual descriptions for every scene.
- Visuals must match the narration.
- Make each visual description suitable for later AI image or video
  generation.
- Include environments, architecture, people, clothing, landscapes,
  historical atmosphere, camera movement, and lighting where appropriate.
- Avoid anachronisms.

SCRIPT REQUIREMENTS:

- Make the story flow naturally from beginning to end.
- Use engaging documentary-style narration.
- Avoid unnecessary repetition.
- Make scene durations add up approximately to the requested
  {req.duration_minutes}-minute target.
- Keep the script suitable for professional YouTube narration.
- Do not mention these instructions in the final script.
"""

    try:

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
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


@app.post("/api/voice")
def generate_voice(req: VoiceRequest):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )

    try:

        client = genai.Client(api_key=api_key)

        voice_prompt = f"""
Read the following documentary narration in a professional,
cinematic YouTube documentary voice.

Language: {req.language}

Voice style:
- Natural
- Clear
- Warm
- Authoritative
- Cinematic
- Moderate speaking pace
- Appropriate dramatic emphasis
- No background music
- Do not add words that are not in the supplied narration

Narration:

{req.text}
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
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "voice": "Kore"
                    }
                }
            },
        )

        audio_data = None

        if response.candidates:
            candidate = response.candidates[0]

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "inline_data", None):
                        audio_data = part.inline_data.data
                        break

        if not audio_data:
            raise RuntimeError("Gemini returned no audio.")

        if isinstance(audio_data, str):
            audio_bytes = base64.b64decode(audio_data)
        else:
            audio_bytes = audio_data

        filename = "voiceover.wav"
        output_path = AUDIO_DIR / filename

        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_bytes)

        return {
            "success": True,
            "audio_url": f"/api/audio/{filename}",
            "filename": filename
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/api/audio/{filename}")
def get_audio(filename: str):

    safe_name = Path(filename).name
    path = AUDIO_DIR / safe_name

    if not path.exists() or path.suffix.lower() != ".wav":
        raise HTTPException(
            status_code=404,
            detail="Audio not found."
        )

    return FileResponse(
        path,
        media_type="audio/wav",
        filename=safe_name
    )
