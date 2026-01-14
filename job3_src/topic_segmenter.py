import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from google.genai import types

# import utilities
from utils.common import *
from utils.gemini_generation_utils import generate_structured_output_gemini

# --- Helper Functions ---

def load_modalities(dir_path: str, fusion_spec: str, modality_origin: str = "auto") -> dict:
    """
    Loads modality data based on the fusion specification.
    
    This function acts as a factory, calling specific 'build' functions
    (imported from utils) based on the `fusion_spec` string.
    
    :param dir_path: Path to the video directory.
    :param fusion_spec: String key (e.g., "ocr", "transcript", "mOTS").
    :param modality_origin: Origin of the modality data.
    :return: A dictionary containing the loaded modality data.
    """
    allowed = ["ocr", "transcript", "slides", "mOV",  "mOT", "mOS", "mTS", "mOTS"]
    if fusion_spec not in allowed:
        raise ValueError(f"Unknown modality composition: {fusion_spec}. Allowed: {sorted(allowed)}")

    # Prepare a dummy chapter segment (should denote the whole video) to reuse the existing summary builders
    chapter_segment = [{"section_number": 0, "slide_index_start": None, "slide_index_end": None}]
    
    # Load selected modality composition
    if fusion_spec == "transcript":
        out = build_transcript_blocks_v1(dir_path, modality_origin, chapter_segment)
    else:
        raise ValueError(f"Unknown modality: {fusion_spec}")

    if not out:
        raise RuntimeError("No valid modalities loaded.")
    
    return out['0']

# --- Schema Builders ---

def schema_ocr_cleaning():
    """Returns the schema for the OCR cleaning task."""
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "slide_number": types.Schema(type="INTEGER"),
                "ocr_text": types.Schema(type="STRING"),
            },
            required=["slide_number","ocr_text"],
        ),
    )

def schema_topics_segmentation():
    """Returns the schema for the topic segmentation task."""
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "section_number": types.Schema(type="INTEGER"),
                "section_title": types.Schema(type="STRING"),
                "slide_index_start": types.Schema(type="INTEGER"),
                "slide_index_end": types.Schema(type="INTEGER"),
            },
            required=["section_number","section_title","slide_index_start","slide_index_end"],
        ),
    )

# --- Main Generation Functions ---

def ocr_cleaning_with_gemini(
    video_dir: str,
    input_prompts_dir: str = "input/prompts",
    prompt_file: str = "summary/multimodal_prompt.txt",
) -> Optional[Any]:
    """
    Builds a prompt and calls Gemini to clean OCR text.
    
    :param video_dir: Path to the specific video's directory.
    :param input_prompts_dir: Path to the main prompts directory.
    :param prompt_file: Filename of the prompt template to use.
    :return: The result from the generation function, or None on failure.
    """
    
    # --- Input paths ---
    prompt_path = os.path.join(input_prompts_dir, prompt_file)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    # --- Load OCR ---
    try:
        context = load_modalities(video_dir, mods=["ocr"])
    except Exception as e:
        raise FileNotFoundError(f"Failed to load OCR: {e}")

    # --- Build the textual instruction from template ---
    prompt_text = build_prompt_from_template(
        prompt_path,
        min_words=0,
        max_words=0
    )
    model_input = [prompt_text] + context
    save_filepath = os.path.join(video_dir, f"prompt_for_ocr_text_cleaning.txt")
    save_model_input_as_text(model_input, save_filepath)

    # --- Call generation function ---
    try:
        output_path = os.path.join(video_dir, "ocr_text_cleaned.json")
        result = generate_structured_output_gemini(
            model_input, 
            output_path, 
            schema_ocr_cleaning() 
        )
    except Exception as e:
        print(f"[!] Generation failed: {e}")
        return None

    return result

def topic_segmentation_with_modalities(
    video_dir: str,
    fusion_spec: str,
    modality_origin: str = "auto",
    model_str: str = "gemini-2.0-flash",
    generation_mode: str = "deterministic",
    input_prompts_dir: str = "input/prompts",
    prompt_file: str = "topics/transcript_topics_prompt_v1.txt",
    root_output_dir: str = "output/videos",
    output_filename: str = "topics/transcript_topics_v1.json",
    save_model_input: bool = True,
    override_output: bool = False,
) -> Optional[Any]:
    """
    Builds a multimodal prompt and calls Gemini for topic segmentation.

    :param video_dir: Path to the specific video's directory.
    :param fusion_spec: Modality key (e.g., "transcript", "mOTS").
    :param ... (other params): Configuration for generation.
    :return: None. Output is saved to a file.
    """
    
    # --- Output paths ---
    video_folder = os.path.basename(os.path.normpath(video_dir))
    root_out_dir = os.path.join(root_output_dir, video_folder)
    os.makedirs(root_out_dir, exist_ok=True)
    
    output_path = os.path.join(root_out_dir, output_filename)
    if os.path.exists(output_path) and not override_output:
        print(f"[!] Output file already exists, skipping: {output_path}")
        return None

    # --- Input paths ---
    prompt_path = os.path.join(input_prompts_dir, prompt_file)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    # --- Build the textual instruction from template ---
    prompt_text = build_prompt_from_template(prompt_path)

    # --- Load requested modality composition ---
    try:
        generation_context = load_modalities(video_dir, fusion_spec, modality_origin)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load requested modality composition {fusion_spec}: {e}")


    # Build model input in common_prompt format
    model_input = {"prompt": prompt_text, **generation_context}

    if save_model_input:
        save_filepath = os.path.join(video_dir, os.path.basename(prompt_file))
        save_model_input_as_text(model_input, save_filepath)
    
    # return None
    
    # --- Call generation function ---
    try:
        generate_structured_output_gemini(
            model_input, 
            output_path, 
            schema=schema_topics_segmentation(), 
            model=model_str, 
            mode=generation_mode
        )
    except Exception as e:
        print(f"[!] Generation failed: {e}")
        return None


# ---------------- Main ----------------
if __name__ == "__main__":
    helper = Helper()

    # --- Configure Paths ---
    root_videos_dir = "input/videos"
    dataset = "LPM_Dataset"                 # "M3AV_Dataset" | "VISTA_Dataset" | "LPM_Dataset"
    root_output_dir = "output/videos"

    # --- Get list of video folders to process ---
    dataset_path = os.path.join(root_videos_dir, dataset)
    video_list = helper.get_subfolder_names(dataset_path)
    print(f"Total {len(video_list)} video folders to process: \n{video_list}")

    # --- OCR Data Cleaning (Example) ---
    # ocr_cleaning_with_gemini(
    #     video_dir=video_dir,
    #     prompt_file="preprocess/ocr_cleaning.txt"
    # )

    failed_video_topics = {
        "v3": ['ml1_4', 'anat1_2', 'speaking_5', 'bio1_2', 'speaking_2', 'anat1_5', 'psy1_4'],                                      # -3
        "v4": ['speaking_3', 'speaking_5', 'speaking_4', 'bio1_5', 'anat1_2', 'speaking_2', 'ml1_3', 'ml1_4', 'psy1_4', 'bio1_2'],  # -1
        "v5": ['psy1_2', 'speaking_2', 'psy1_4', 'anat1_2', 'bio1_2', 'ml1_4', 'speaking_5', 'speaking_3'],                         # -2
        "v6": ['bio1_2', 'psy1_4', 'anat1_2', 'ml1_4']                                                                              # -0
        }

    # Collect and sort all unique failed videos
    all_failed_videos = sorted(set(v for vlist in failed_video_topics.values() for v in vlist))
    print(f"Total {len(all_failed_videos)} unique failed videos from previous runs: \n{all_failed_videos}")
    video_list = all_failed_videos

    # --- Topic Segmentation ---

    # Configure Parameters
    modality_composition = "transcript"             # "ocr" | "transcript" | "slides" | "mOV" | "mOS" | "mOT" | "mTS" | "mOTS"
    modality_origin = "dataset"                     # "auto" | "dataset"
    gemini_model_version = "2.5-flash"              # "2.0-flash" | "2.5-flash" | "2.5-flash-lite"
    generation_mode = "restrictive"                 # "deterministic" | "restrictive"
    experiment_version = "v11"                       # custom experiment tag

    # Process each video folder
    for idx, video_folder in enumerate(video_list, start=1):
        # video_folder = "anat1_2"  # For testing single video
        video_dir = os.path.join(root_videos_dir, dataset, video_folder)
        print(f"\nNow processing [{idx}/{len(video_list)}] : {video_folder}")

        topic_segmentation_with_modalities(
            video_dir = video_dir,
            fusion_spec = modality_composition,
            modality_origin = modality_origin,
            model_str = f"gemini-{gemini_model_version}",
            generation_mode = generation_mode, 
            prompt_file = f"topics/{modality_composition}_topics_prompt_{experiment_version}.txt",
            output_filename = f"topics/{modality_composition}_topics_{experiment_version}.json",
            root_output_dir = os.path.join(root_output_dir, dataset),
            override_output=True
        )
        # break  # Remove this break to process all videos
