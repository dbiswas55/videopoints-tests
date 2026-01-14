import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from google.genai import types

# import utilities
from utils.common import *
from job3_src.io_utils import *
from utils.gemini_generation_utils import generate_structured_output_gemini

# --- Helper Functions ---

def read_prompt_template(filepath: str) -> str:
    """
    Reads a prompt template file and returns its content.
    
    :param filepath: Path to the prompt template file.
    :return: The template content as a string.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        template = f.read()
    return template.strip() + "\n"

def load_modalities(dir_path: str, fusion_spec: str = None) -> dict:
    """
    Loads modality data based on the fusion specification.
    
    This function acts as a factory, calling specific 'build' functions
    (imported from utils) based on the `fusion_spec` string.
    
    :param dir_path: Path to the video directory.
    :param fusion_spec: String key (e.g., "ocr", "transcript", "mOTS").
    :return: A dictionary containing the loaded modality data.
    """
    allowed = ["transcript"]
    if fusion_spec not in allowed:
        raise ValueError(f"Unknown modality composition: {fusion_spec}. Allowed: {sorted(allowed)}")

    # Load selected modality composition
    if fusion_spec == "transcript":
        out = build_transcript_blocks_v1(dir_path)
    else:
        raise ValueError(f"Unknown modality: {fusion_spec}")

    if not out: raise RuntimeError("No valid modalities loaded.")
    
    return out

# --- Schema Builders ---

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

def topic_segmentation_with_modalities(
    video_dir: str,
    fusion_spec: str = "transcript",
    model_str: str = "gemini-2.0-flash",
    generation_mode: str = "deterministic",
    prompt_file: str = "toc_generation_prompt_v1.txt",
    output_filename: str = "topics_v1.json",
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
    output_path = os.path.join(video_dir, output_filename)
    if os.path.exists(output_path) and not override_output:
        print(f"[!] Output file already exists, skipping: {output_path}")
        return None

    # --- Input paths ---
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_file)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    # --- Build the textual instruction from template ---
    prompt_text = read_prompt_template(prompt_path)

    # --- Load requested modality composition ---
    try:
        generation_context = load_modalities(video_dir, fusion_spec)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load requested modality composition {fusion_spec}: {e}")


    # Build model input in common_prompt format
    model_input = {"prompt": prompt_text, **generation_context}

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

    # Configure file and directory paths
    video_path = "output/web_vpp/v8020/BIOL2301_lecture05_0609_8020.mp4"
    video_dir = os.path.dirname(video_path)


    # --- Topic Segmentation ---

    # Configure Parameters
    modality_composition = "transcript"             # "ocr" | "transcript" | "slides" | "mOV" | "mOS" | "mOT" | "mTS" | "mOTS"
    gemini_model_version = "2.5-flash"              # "2.0-flash" | "2.5-flash" | "2.5-flash-lite"
    generation_mode = "restrictive"                 # "deterministic" | "restrictive"
    experiment_version = "v1"                       # custom experiment tag


    topic_segmentation_with_modalities(
        video_dir = video_dir,
        fusion_spec = modality_composition,
        model_str = f"gemini-{gemini_model_version}",
        generation_mode = generation_mode, 
        prompt_file = f"toc_generation_prompt_{experiment_version}.txt",
        output_filename = f"topics_{experiment_version}.json",
        override_output=True
    )
