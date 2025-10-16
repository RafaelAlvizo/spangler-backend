from contextlib import asynccontextmanager
import logging
import os
from typing import Annotated, Optional
import aiohttp
from fastapi import BackgroundTasks, Body, FastAPI, Response
from fastapi.responses import FileResponse
import uvicorn

import asyncio

from openai import OpenAI
from cyksuid.v2 import ksuid

TEMP_PATH = "temp"
session: Optional[aiohttp.ClientSession] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    session = aiohttp.ClientSession()
    yield
    await session.close()


app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(
        "spangler:app",
        host="0.0.0.0",
        # workers=1,
        port=8765,
    )


@app.post("/generate_without_save", response_class=FileResponse)
async def generate_without_save(
    word: Annotated[str, Body()],
    background_tasks: BackgroundTasks,
    voice: Annotated[str, Body()] = "alloy",
):
    print(f"Received {word}")
    mp3 = await openai_text_to_speech(word, voice)
    if not mp3:
        return Response(content="No audio generated", status_code=403)

    background_tasks.add_task(os.remove, mp3)
    return FileResponse(mp3, media_type="audio/mp3")


client = OpenAI()


async def openai_text_to_speech(text: str, voice: str = "alloy") -> str:
    file_path = f"{TEMP_PATH}/openai/{str(ksuid())}.mp3"  # type: ignore

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions="""
Accent/Affect: Warm, refined, and gently instructive, reminiscent of a friendly art instructor.

Tone: Calm, encouraging, and articulate, clearly describing each step with patience.

Pacing: Normal speed

Emotion: Cheerful, supportive, and pleasantly enthusiastic; convey genuine enjoyment and appreciation of art/literature.

Pronunciation: Pronounce all the Spanish words in the Mexican accent while keeping the English words in American accent

Personality Affect: Friendly and approachable with a hint of sophistication; speak confidently and reassuringly, guiding users through each step patiently and warmly.
        """,
    ) as response:
        response.stream_to_file(file_path)

    return file_path


def test_openai_tts():
    result = asyncio.run(openai_text_to_speech("Hello, world!", "ash"))
    print(result)
    os.remove(result)
