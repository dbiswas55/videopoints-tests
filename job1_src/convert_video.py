import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import subprocess
import whisper
from utils.common import *



def get_video_info(path: str):
    """Extracts key audio and video parameters from a file or folder using ffprobe."""

    video_path = helper.get_video_path(path)
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


def crop_video_ffmpeg(video_filepath: str, start_time: float, end_time: float, output_path=None):
    """
    Crop a video using ffmpeg from start_time to end_time (in seconds).

    Parameters:
        video_filepath (str): Path to the input video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_path (str): Output filepath to save the cropped video.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    
    duration = end_time - start_time
    if output_path is None:
        base, _ = os.path.splitext(video_filepath)
        output_path = f"{base}_{duration//60}min.mp4"

    command = [
        'ffmpeg',
        '-y',                     # Overwrite output file if it exists
        '-ss', str(start_time),  # Start time
        '-i', video_filepath,        # Input file
        '-t', str(duration),     # Duration of the clip
        '-c', 'copy',            # Copy codec (no re-encoding)
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Cropped video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error cropping video:", e)
    return output_path


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

def transcribe_audio_whisper(audio_filepath, model_size="medium", language="en", initial_prompt=None, save_path=None):
    """
    Run OpenAI Whisper (PyTorch) and save a minimal JSON:
      { "text": <full transcript>,
        "segments": [ {id, start, end, text}, ... ] }
    """
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

    return clean


def video_processing_pipeline(
    path: str,
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
        path (str): Path to the input video file or video folder path with single video file.
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
    video_path = helper.get_video_path(path)
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


if __name__ == "__main__":
    helper = Helper()
    
    # --- Configure Paths ---
    root_videos_dir = "data/videos"
    dataset = "LPM_Dataset"                               # "M3AV_Dataset" | "VISTA_Dataset" | "YouTubes" | "LPM_Dataset"
    root_output_dir = "output/modality_extraction"

    # Setting up directories
    dataset_path = os.path.join(root_videos_dir, dataset)
    output_dir = os.path.join(root_output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)


    # Get list of video folders to process
    selected_videos_json = os.path.join(dataset_path, f"selected_videos_{dataset.lower()}.json")
    video_list = helper.read_from_json(selected_videos_json)
    print(f"Total {len(video_list)} video folders to process: \n{video_list}")

    start_time = time.time()

    for idx, video_folder in enumerate(video_list):
        # video_folder = "CMU_MML_L2"  # For testing single video [also uncomment break]; comment this line to process all videos
        print(f"\n\n[{idx+1}/{len(video_list)}] Processing video folder: {video_folder}")
        video_folderpath = os.path.join(dataset_path, video_folder)
        # get_video_info(video_folderpath)

        video_processing_pipeline(
            video_folderpath, apply_convert=True, 
            apply_extract_audio=True, 
            apply_transcribe=True, 
            skip_convert_on_existing=True,
            fps=10, 
            target_height=720, 
            whisper_model='small',
            output_dir=os.path.join(output_dir, video_folder),
        )
        break  # Remove this break to process all videos


    print(f"\n✅ All steps completed in {time.time() - start_time:.2f} seconds.")
