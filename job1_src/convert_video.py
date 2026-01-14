import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import subprocess
import whisper



def get_video_info(video_path: str):
    """Extracts key audio and video parameters from a file or folder using ffprobe."""
    start_time = time.time()
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    metadata = json.loads(result.stdout)

    video_info = {}
    audio_info = {}

    for stream in metadata["streams"]:
        if stream["codec_type"] == "video":
            video_info = {
                "Codec": stream.get("codec_name"),
                "Resolution": f"{stream.get('width')}x{stream.get('height')}",
                "Frame Rate": eval(stream.get("r_frame_rate", "0/1")),
                "Duration (s)": float(stream.get("duration", metadata["format"].get("duration", 0))),
                "Total Frames": int(stream.get("nb_frames", "0")),
                "Bitrate": stream.get("bit_rate", "N/A")
            }

        elif stream["codec_type"] == "audio":
            audio_info = {
                "Codec": stream.get("codec_name"),
                "Sample Rate": stream.get("sample_rate"),
                "Channels": stream.get("channels"),
                "Channel Layout": stream.get("channel_layout", "N/A"),
                "Bitrate": stream.get("bit_rate", "N/A"),
            }

    print("Video Info:")
    for k, v in video_info.items():
        print(f"  {k}: {v}")

    print("\nAudio Info:")
    for k, v in audio_info.items():
        print(f"  {k}: {v}")
    
    print(f"⏱️  get_video_info completed in {time.time() - start_time:.2f}s")


def convert_video_to_h264(
    video_filepath,
    fps=10,
    target_height=720,
    crf=20,
    preset="medium",
    strip_audio=False,
    skip_on_existing=True,
    output_dir=None
):
    """
    Convert a video to H.264 format, resizing by fixed height while maintaining aspect ratio,
    and optionally enforcing frame rate and audio stripping.

    Parameters:
        video_filepath (str): Path to input video file.
        fps (int): Target frames per second for conversion.
        target_height (int): Output video height (aspect ratio preserved).
        crf (int): Constant Rate Factor (quality). Lower = higher quality.
        preset (str): Encoding speed-efficiency tradeoff.
        strip_audio (bool): If True, removes audio during conversion.
        skip_on_existing (bool): If True, skip conversion if converted file already exists (contains "_fps").
        output_path (str): Optional output directory.

    """
    start_time = time.time()
    
    if not os.path.exists(video_filepath):
        raise FileNotFoundError(f"Input file not found: {video_filepath}")

    if output_dir is None:
        base, _ = os.path.splitext(video_filepath)
        output_path = f"{base}_fps{fps}.mp4"
    else:
        filename, ext = os.path.splitext(os.path.basename(video_filepath))
        output_path = os.path.join(output_dir, f"{filename}_fps{fps}.mp4")

    # Skip if input already looks like a converted file
    if skip_on_existing and os.path.exists(output_path):
        print(f"[INFO] Skipping video conversion as it already exists at {output_path}.")
        return output_path
    
    print(f"[DEBUG] Converting '{os.path.basename(video_filepath)}' to H.264 with target height={target_height}, fps={fps}, crf={crf}, preset={preset}, strip_audio={strip_audio}")

    # FFmpeg filter to maintain aspect ratio using fixed height
    scale_filter = f"scale=-2:{target_height}"

    # Combine scale and fps filters
    filters = f"fps={fps},{scale_filter}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_filepath,
        "-vf", filters,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p"  # Standard pixel format
    ]

    if strip_audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd.append(output_path)

    print("🚀 Running FFmpeg command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Conversion complete: {output_path}")
    print(f"⏱️  convert_video_to_h264 completed in {time.time() - start_time:.2f}s")
    return output_path


def extract_mp3_from_video(video_path, bitrate='64k', output_path=None):
    """
    Extracts MP3 audio from a video file using FFmpeg.

    Args:
        video_path (str): Path to the input video file.
        bitrate (str): Audio bitrate for MP3 output (default is '64k').
        output_path (str, optional): Path to save the MP3 file. If None, replaces video extension with '.mp3'.

    Returns:
        str: Path to the extracted MP3 file.
    """
    start_time = time.time()
    
    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}.mp3"

    # Run FFmpeg to extract audio
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-b:a', bitrate,
        output_path
    ]

    subprocess.run(cmd, stderr=subprocess.PIPE)
    print(f"⏱️  extract_mp3_from_video completed in {time.time() - start_time:.2f}s")


def transcribe_audio_whisper(audio_filepath, model_size="medium", language="en", initial_prompt=None, save_path=None):
    """
    Run OpenAI Whisper (PyTorch) and save a minimal JSON:
      { "text": <full transcript>,
        "segments": [ {id, start, end, text}, ... ] }
    """
    start_time = time.time()
    
    if not os.path.isfile(audio_filepath):
        raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_filepath}")
    res = model.transcribe(
        audio_filepath,
        language=language,
        initial_prompt=initial_prompt,
        fp16=False,
        verbose=False
    )

    # Build minimal structure
    clean = {
        "text": res.get("text", ""),
        "segments": [
            {
                "id": int(seg.get("id", i)),
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": (seg.get("text") or "").strip()
            }
            for i, seg in enumerate(res.get("segments", []))
        ]
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f"Transcript JSON saved to: {save_path}")
    
    print(f"⏱️  transcribe_audio_whisper completed in {time.time() - start_time:.2f}s")
    return clean


def video_processing_pipeline(
    video_path: str,
    apply_convert: bool = True,
    apply_extract_audio: bool = True,
    apply_transcribe: bool = True,
    skip_convert_on_existing=True,
    fps: int = 10,
    target_height: int = 720,
    strip_audio: bool = False,
    audio_bitrate: str = "64k",
    whisper_model: str = "small",
    output_dir = None,
    create_subject_folder: bool = False
):
    """
    End-to-end video processing pipeline:
      1. Convert video to H.264
      2. Extract MP3 audio
      3. Transcribe audio with Whisper

    Parameters:
        video_path (str): Path to the input video file.
        apply_convert (bool): Whether to convert video to H.264.
        apply_extract_audio (bool): Whether to extract audio as MP3.
        apply_transcribe (bool): Whether to run Whisper transcription.
        skip_convert_on_existing (bool): Skip conversion if already converted or file exists.
        fps (int): Target frames per second for conversion.
        target_height (int): Target video height (maintains aspect ratio).
        strip_audio (bool): If True, removes audio during conversion.
        audio_bitrate (str): MP3 audio bitrate for extraction.
        whisper_model (str): Whisper model size (tiny, small, medium, large).
        output_dir (str): Output directory to save

    Returns:
        dict | None: Transcription result if `apply_transcribe` is True, else None.
    """
    pipeline_start = time.time()
    
    if create_subject_folder:
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_dir, filename_no_ext)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert video
    if apply_convert:
        print("\n[STEP 1] Converting video...")
        convert_video_to_h264(
            video_filepath=video_path,
            fps=fps,
            target_height=target_height,
            strip_audio=strip_audio,
            skip_on_existing=skip_convert_on_existing,
            output_dir=output_dir                              # this line is different from the original source
        )

    # Step 2: Extract MP3
    audio_path = f"{os.path.splitext(video_path)[0]}.mp3"
    if not apply_extract_audio:
        print(f"[INFO] Audio extraction is disabled.")
    elif skip_convert_on_existing and os.path.exists(audio_path):
        print(f"[INFO] Skipping audio extraction as it already exists.")
    else:
        print("\n[STEP 2] Extracting audio...")
        extract_mp3_from_video(video_path, bitrate=audio_bitrate)
        print(f"[OK] Audio extracted at {audio_path}")

    # Step 3: Transcribe audio
    json_path = os.path.join(output_dir, f"transcript_text.json")
    if not apply_transcribe:
        print(f"[INFO] Transcription is disabled.")
    elif skip_convert_on_existing and os.path.exists(json_path):
        print(f"[INFO] Skipping transcription as it already exists.")
    else:
        print("\n[STEP 3] Transcribing audio...")
        transcribe_audio_whisper(audio_path, model_size=whisper_model, save_path=json_path)
        print("[OK] Transcription completed.")
    
    print(f"\n⏱️  video_processing_pipeline completed in {time.time() - pipeline_start:.2f}s")



if __name__ == "__main__":

    # Configure file and directory paths
    video_path = "output/web_vpp/v7158/Video_2550_7158_testing.mp4"

    # get_video_info(video_path)
    


    video_processing_pipeline(
        video_path, 
        apply_convert=True, 
        apply_extract_audio=True, 
        apply_transcribe=True, 
        skip_convert_on_existing=True,
        fps=10, 
        target_height=720, 
        whisper_model='small',
        output_dir=os.path.dirname(video_path)
    )
