# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import uvicorn
import os
import uuid
import asyncio
from video_pipeline import (
    generate_movie_from_script,
    generate_character_clip_from_image,
    check_ffmpeg,
)

app = FastAPI(title="HD AI Movie Creator Backend")

# CORS - open for development; change to your frontend origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure ffmpeg exists on startup
@app.on_event("startup")
async def startup_event():
    ok, msg = check_ffmpeg()
    if not ok:
        print("WARNING:", msg)


@app.get("/")
async def root():
    return {"status": "Video Creator API is running!"}


@app.post("/generate_movie")
async def generate_movie(
    script: str = Form(...),
    title: Optional[str] = Form("movie"),
    target_fps: Optional[int] = Form(24),
    target_resolution: Optional[str] = Form("1920x1080"),
    images: Optional[List[UploadFile]] = File(None),
    clips: Optional[List[UploadFile]] = File(None),
    voice: Optional[str] = Form("default"),
    realtime_character: Optional[bool] = Form(False),
):
    """
    Generate a movie. Accepts:
     - script (text)
     - optional uploaded images and clips (multiple)
    Returns JSON { status, file } where file is filename stored on server root.
    """
    job_id = str(uuid.uuid4())
    workdir = os.path.join("jobs", job_id)
    os.makedirs(workdir, exist_ok=True)

    saved_images = []
    saved_clips = []

    try:
        # save uploads
        if images:
            for img in images:
                path = os.path.join(workdir, img.filename)
                with open(path, "wb") as f:
                    f.write(await img.read())
                saved_images.append(path)

        if clips:
            for c in clips:
                path = os.path.join(workdir, c.filename)
                with open(path, "wb") as f:
                    f.write(await c.read())
                saved_clips.append(path)

        # run generation (blocking — for production enqueue to worker)
        out_filename = f"{title.replace(' ', '_')}_{job_id}.mp4"
        out_path = os.path.abspath(out_filename)

        generate_movie_from_script(
            script=script,
            output_path=out_path,
            images=saved_images,
            clips=saved_clips,
            voice_profile=voice,
            fps=int(target_fps),
            resolution=target_resolution,
            realtime_character=bool(realtime_character),
            workdir=workdir,
        )

        return {"status": "done", "file": out_filename}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/character_from_image")
async def character_from_image(file: UploadFile = File(...), name: str = Form("character")):
    """
    Generate a short animated character clip from a single image.
    """
    workdir = os.path.join("jobs", str(uuid.uuid4()))
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    out_path = generate_character_clip_from_image(path, name=name, duration=8, resolution="1280x720", workdir=workdir)
    return {"status": "done", "file": out_path}


@app.get("/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(filename, media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
