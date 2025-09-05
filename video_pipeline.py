# video_pipeline.py
"""
Video pipeline using ffmpeg + Pillow + gTTS.

Key functions:
 - generate_movie_from_script(...)
 - generate_character_clip_from_image(...)

This implementation:
 - Parses script into scenes (split by blank lines)
 - For each scene: create TTS audio, build a scene video from an image or uploaded clip or placeholder background
 - Uses ffmpeg to encode reliable HD .mp4 files (libx264 + aac)
 - Concatenates scene files safely using ffmpeg concat demuxer
"""

import os
import tempfile
import subprocess
import uuid
import math
import time
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import shutil

# Helper: verify ffmpeg available
def check_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return True, f"ffmpeg at {ffmpeg}"
    else:
        return False, "ffmpeg not found in PATH. Please ensure ffmpeg is installed on the host."

# --- Script parsing ---
def parse_script_to_scenes(script: str):
    # Split on double newlines or '---' markers
    parts = []
    for chunk in script.split("\n\n"):
        c = chunk.strip()
        if c:
            parts.append(c)
    # fallback if empty
    if not parts:
        parts = ["<empty scene>"]
    # estimate duration per scene based on words (approx 150 wpm)
    scenes = []
    for p in parts:
        words = len(p.split())
        dur = max(4, int(words / 2.5))  # ~2.5 words/sec => 150 wpm
        scenes.append({"text": p, "duration": dur})
    return scenes

# --- TTS ---
def synthesize_tts(text: str, out_path: str, lang="en"):
    # gTTS requires network; replace with other TTS if desired
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path

# --- Image slide generation ---
def make_text_image(text: str, size=(1920,1080), bg_color=(10,10,30), fg_color=(230,230,230), fontsize=48):
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    # choose a fallback font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    # wrap text
    max_width = size[0] - 200
    lines = []
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        wbox = draw.textbbox((0,0), test, font=font)
        if wbox[2] > max_width and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    # compute text block height
    line_h = font.getsize("A")[1] + 10
    block_h = line_h * len(lines)
    y = (size[1] - block_h) // 2
    for ln in lines:
        wbox = draw.textbbox((0,0), ln, font=font)
        x = (size[0] - (wbox[2] - wbox[0])) // 2
        draw.text((x, y), ln, font=font, fill=fg_color)
        y += line_h
    return img

def save_pil_image(img: Image.Image, out_path: str):
    img.save(out_path, format="PNG")
    return out_path

# --- Create scene video from single image using ffmpeg (loop image for duration) ---
def image_to_video(image_path: str, out_video: str, duration: int = 6, resolution="1920x1080", fps=24):
    """
    Creates an H264 MP4 video from an image with given duration.
    Uses ffmpeg -loop 1 -i image -c:v libx264 -t duration -pix_fmt yuv420p -vf scale
    """
    w,h = map(int, resolution.split("x"))
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p",
        "-vf", f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
        "-r", str(fps),
        out_video
    ]
    subprocess.run(cmd, check=True)
    return out_video

# --- Trim or loop uploaded video to desired duration (create a clip) ---
def clip_to_duration(input_clip: str, out_clip: str, duration: int, fps=24, resolution="1920x1080"):
    w,h = map(int, resolution.split("x"))
    # Get duration using ffprobe
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_clip],
            check=True, capture_output=True, text=True
        )
        src_dur = float(probe.stdout.strip())
    except Exception:
        src_dur = duration

    if src_dur >= duration:
        # trim
        cmd = [
            "ffmpeg", "-y", "-i", input_clip,
            "-ss", "0", "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(fps),
            out_clip
        ]
    else:
        # loop to fill
        # create concat list by repeating the same file
        repeats = math.ceil(duration / max(0.1, src_dur))
        tmp_list = []
        tmp_files = []
        for i in range(repeats):
            tmp_files.append(input_clip)
        # create intermediate concatenated file using ffmpeg -stream_loop
        cmd = [
            "ffmpeg", "-y", "-stream_loop", str(repeats-1), "-i", input_clip,
            "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p",
            "-vf", f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(fps),
            out_clip
        ]
    subprocess.run(cmd, check=True)
    return out_clip

# --- Merge video file with audio (replace audio) ---
def mux_video_audio(video_in: str, audio_in: str, out_file: str):
    # Use shortest to avoid audio longer than video
    cmd = [
        "ffmpeg", "-y", "-i", video_in, "-i", audio_in,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        out_file
    ]
    subprocess.run(cmd, check=True)
    return out_file

# --- Concatenate multiple video parts (ffmpeg concat demuxer) ---
def concat_videos(video_paths: List[str], out_path: str):
    if len(video_paths) == 1:
        # copy single file
        shutil.copy(video_paths[0], out_path)
        return out_path

    list_file = os.path.join(os.path.dirname(out_path), f"concat_{uuid.uuid4().hex}.txt")
    with open(list_file, "w") as f:
        for p in video_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path]
    subprocess.run(cmd, check=True)
    try:
        os.remove(list_file)
    except:
        pass
    return out_path

# --- High-level scene render ---
def render_scene(scene_text: str, scene_assets: List[str], voice_profile: str,
                 duration_override: Optional[int], resolution: str, fps: int, workdir: str):
    """
    Render a single scene: returns path to scene video (with audio baked).
    """
    duration = duration_override or max(4, int(len(scene_text.split()) / 2.5))
    # Prepare TTS audio for this scene
    audio_path = os.path.join(workdir, f"tts_{uuid.uuid4().hex}.mp3")
    synthesize_tts(scene_text, audio_path)

    # pick asset (image first, then clip)
    chosen = None
    if scene_assets:
        for a in scene_assets:
            ext = os.path.splitext(a)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                # create a resized image copy
                chosen = ("image", a)
                break
            elif ext in [".mp4", ".mov", ".webm", ".mkv", ".avi"]:
                chosen = ("clip", a)
                break

    tmp_scene_video = os.path.join(workdir, f"scene_{uuid.uuid4().hex}.mp4")
    if chosen and chosen[0] == "image":
        # create a slide image (with text overlay optionally)
        # For better effect, create text image with scene text summary as fallback subtitle
        text_img = make_text_image(scene_text, size=tuple(map(int, resolution.split("x"))))
        img_path = os.path.join(workdir, f"scene_img_{uuid.uuid4().hex}.png")
        save_pil_image(text_img, img_path)
        # create video from uploaded image (use uploaded image covering resolution) - prefer uploaded asset if exists
        # try using uploaded image directly if it exists in scene_assets
        try:
            # use the uploaded image path instead of generated text image to show character if provided
            if scene_assets and os.path.exists(scene_assets[0]):
                base_image = scene_assets[0]
            else:
                base_image = img_path
        except:
            base_image = img_path
        image_to_video(base_image, tmp_scene_video, duration=duration, resolution=resolution, fps=fps)
    elif chosen and chosen[0] == "clip":
        clip_to_duration(chosen[1], tmp_scene_video, duration=duration, fps=fps, resolution=resolution)
    else:
        # no asset -> create text background
        text_img = make_text_image(scene_text, size=tuple(map(int, resolution.split("x"))))
        img_path = os.path.join(workdir, f"scene_img_{uuid.uuid4().hex}.png")
        save_pil_image(text_img, img_path)
        image_to_video(img_path, tmp_scene_video, duration=duration, resolution=resolution, fps=fps)

    # now mux TTS audio into the scene video
    tmp_scene_with_audio = os.path.join(workdir, f"scene_a_{uuid.uuid4().hex}.mp4")
    mux_video_audio(tmp_scene_video, audio_path, tmp_scene_with_audio)

    # cleanup scene files (keep final)
    try:
        os.remove(tmp_scene_video)
        os.remove(audio_path)
    except:
        pass

    return tmp_scene_with_audio

# --- Character clip generator (simple Ken Burns + text) ---
def generate_character_clip_from_image(image_path: str, name: str = "char", duration: int = 6, resolution: str = "1280x720", workdir: Optional[str] = None):
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="char_")
    out_video = os.path.abspath(f"{name}_{uuid.uuid4().hex}.mp4")
    # create ken burns style: resize + slight zoom simulated by scaling (ffmpeg filter could do zoompan but keep simple)
    # For simplicity, just create image->video
    image_to_video(image_path, out_video, duration=duration, resolution=resolution, fps=24)
    return out_video

# --- Main orchestrator ---
def generate_movie_from_script(
    script: str,
    output_path: str = "movie_out.mp4",
    images: Optional[List[str]] = None,
    clips: Optional[List[str]] = None,
    voice_profile: str = "default",
    fps: int = 24,
    resolution: str = "1920x1080",
    realtime_character: bool = False,
    workdir: Optional[str] = None,
):
    images = images or []
    clips = clips or []
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="movie_")
    os.makedirs(workdir, exist_ok=True)

    ff_ok, msg = check_ffmpeg()
    if not ff_ok:
        raise RuntimeError(msg)

    scenes = parse_script_to_scenes(script)
    scene_files = []

    for idx, sc in enumerate(scenes):
        assets_for_scene = []
        # simple round-robin of user assets
        if images:
            assets_for_scene.append(images[idx % len(images)])
        if clips:
            assets_for_scene.append(clips[idx % len(clips)])

        scene_file = render_scene(
            scene_text=sc["text"],
            scene_assets=assets_for_scene,
            voice_profile=voice_profile,
            duration_override=sc.get("duration"),
            resolution=resolution,
            fps=fps,
            workdir=workdir,
        )
        scene_files.append(scene_file)

    # concatenate scenes
    concat_out = os.path.abspath(output_path)
    concat_videos(scene_files, concat_out)

    # cleanup intermediate scene files
    for f in scene_files:
        try:
            os.remove(f)
        except:
            pass

    return concat_out
