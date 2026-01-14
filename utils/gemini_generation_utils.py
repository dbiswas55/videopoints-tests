#!/usr/bin/env python3
"""
Gemini API Utilities for Content Generation

This module provides a unified interface for generating text and structured JSON outputs
using Google's Gemini models. It supports both single and batch generation modes.

Common Prompt Format:
    All generation functions accept a common_prompt dict with the following structure:
    {
        "system": str | None,           # Optional system instruction
        "prompt": str,                  # Required primary instruction/question
        "blocks": List[Dict],           # List of content blocks (text/images)
    }

    Block types:
        - {"type": "text", "text": "..."}
        - {"type": "image", "path": "...", "mime_type": "image/png"}

Dependencies:
    pip install -U google-genai python-dotenv
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types



# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

GENERATION_MODES: Dict[str, Dict[str, float]] = {
    "deterministic": {
        "temperature": 0.0,  # No randomness - always choose most likely token
        "top_k": 1,          # Only consider top-1 candidate
        "top_p": 1.0,        # Full probability mass
    },
    "restrictive": {
        "temperature": 0.3,  # Low randomness for controlled variation
        "top_k": 40,         # Default top-k sampling
        "top_p": 0.85,       # Stricter than Gemini default (0.95)
    },
}
"""
Generation modes for controlling model output randomness.

Gemini defaults: temperature=1.0, top_k=40, top_p=0.95
"""

# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

def topic_segmentation_schema() -> genai_types.Schema:
    """
    Default schema for topic/section segmentation tasks.
    
    Returns array of sections with:
        - section_number: int
        - section_title: str
        - slide_index_start: int
        - slide_index_end: int
    """
    return genai_types.Schema(
        type="ARRAY",
        items=genai_types.Schema(
            type="OBJECT",
            properties={
                "section_number": genai_types.Schema(type="INTEGER"),
                "section_title": genai_types.Schema(type="STRING"),
                "slide_index_start": genai_types.Schema(type="INTEGER"),
                "slide_index_end": genai_types.Schema(type="INTEGER"),
            },
            required=[
                "section_number",
                "section_title",
                "slide_index_start",
                "slide_index_end",
            ],
        ),
    )


# =============================================================================
# CORE HELPER FUNCTIONS
# =============================================================================

def get_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initialize and return a Gemini API client.
    
    Args:
        api_key: Optional API key. If not provided, reads from GEMINI_API_KEY env var.
        
    Returns:
        Configured Gemini client instance.
        
    Raises:
        RuntimeError: If no API key is found.
    """
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing API key: provide api_key parameter or set GEMINI_API_KEY environment variable"
        )
    return genai.Client(api_key=key)


def get_thinking_budget_for(model: str, thinking_budget: Optional[int] = None) -> Optional[int]:
    """
    Determine the appropriate thinking budget for a given model.
    
    Args:
        model: Model identifier (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
        thinking_budget: User-specified budget (if provided, returns as-is)
        
    Returns:
        Thinking budget value:
            - User-provided value if specified
            - -1 for gemini-2.5-pro (thinking cannot be disabled)
            - 0 for gemini-2.5-flash (thinking disabled by default)
            - None for older models (no thinking support)
    """
    if thinking_budget is not None:
        return thinking_budget

    if model.startswith("gemini-2.5-pro"):
        return -1  # Pro requires thinking; -1 means unlimited
    elif model.startswith("gemini-2.5-flash"):
        return 0  # Flash can disable thinking
    else:
        return None  # Older models don't support thinking


def _build_config(
    base_cfg: Dict[str, Any],
    final_budget: Optional[int] = None,
    **extra
) -> genai_types.GenerateContentConfig:
    """
    Build a GenerateContentConfig with base parameters and optional thinking budget.
    
    Args:
        base_cfg: Base configuration dict (temperature, top_k, top_p)
        final_budget: Optional thinking budget for extended reasoning
        **extra: Additional config params (response_mime_type, response_schema, system_instruction)
        
    Returns:
        Configured GenerateContentConfig instance.
    """
    if final_budget is None:
        return genai_types.GenerateContentConfig(**extra, **base_cfg)
    
    return genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=final_budget),
        **base_cfg,
        **extra,
    )


def common_prompt_to_gemini_contents(
    common_prompt: Dict[str, Any],
) -> Tuple[Optional[str], List[genai_types.Content]]:
    """
    Convert common_prompt format to Gemini API-compatible contents.

    Args:
        common_prompt: Dict with keys:
            - system (str | None): Optional system instruction
            - prompt (str): Required primary instruction/question
            - blocks (List[Dict]): Content blocks (text/images)
                Text block: {"type": "text", "text": "..."}
                Image block: {"type": "image", "path": "...", "mime_type": "image/png"}

    Returns:
        Tuple of (system_instruction, contents):
            - system_instruction: Optional string for system parameter
            - contents: List of Content objects for Gemini API
            
    Raises:
        FileNotFoundError: If an image path doesn't exist
    """
    system_instruction = common_prompt.get("system") or None
    prompt_text = common_prompt.get("prompt")
    blocks = common_prompt.get("blocks", [])

    # Validate required fields
    if not prompt_text:
        raise ValueError("common_prompt must contain a non-empty 'prompt' field")

    parts: List[genai_types.Part] = []

    # Add primary prompt/instruction as first text part
    parts.append(genai_types.Part.from_text(text=prompt_text))

    # Process content blocks in order
    for block in blocks:
        block_type = block.get("type")

        if block_type == "text":
            text_content = block.get("text", "")
            if text_content:
                parts.append(genai_types.Part.from_text(text=text_content))

        elif block_type == "image":
            path = block.get("path")
            mime_type = block.get("mime_type")
            
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Image path missing or not found: {path}")
            
            try:
                with open(path, "rb") as f:
                    img_bytes = f.read()
                parts.append(genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            except Exception as e:
                print(f"[Warning] Failed to load image '{path}': {e}")

        else:
            print(f"[Warning] Unknown block type '{block_type}' - skipping")

    contents = [genai_types.Content(role="user", parts=parts)]
    return system_instruction, contents


# =============================================================================
# SINGLE GENERATION FUNCTIONS
# =============================================================================
def generate_text_output_gemini(
    common_prompt: Dict[str, Any],
    save_path: str,
    model: str = "gemini-2.5-flash-lite",
    mode: str = "restrictive",
    thinking_budget: Optional[int] = None,
) -> Optional[str]:
    if mode not in GENERATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(GENERATION_MODES.keys())}")

    # Ensure output directory exists and get client
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    # Convert common_prompt to Gemini contents
    system_instruction, contents = common_prompt_to_gemini_contents(common_prompt)

    # Build config
    final_budget = get_thinking_budget_for(model, thinking_budget)
    base_cfg = dict(**GENERATION_MODES[mode])
    config = _build_config(base_cfg, final_budget, system_instruction=system_instruction)

    # Call Gemini API
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        text_out = resp.text or ""
    except Exception as e:
        print(f"[Error] Model call failed: {e}")
        return None

    # Save output
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_out)
        print(f"[Text:{mode}] Saved output to {save_path}")
    except Exception as e:
        print(f"[Error] Failed to save text output: {e}")
        return None

    return text_out


def generate_structured_output_gemini(
    common_prompt: Dict[str, Any],
    save_path: str,
    schema: Optional[genai_types.Schema] = None,
    model: str = "gemini-2.0-flash",
    mode: str = "restrictive",
    thinking_budget: Optional[int] = None,
) -> Optional[Dict | List]:
    """
    Generate structured JSON output.
    If schema is None, defaults to `topic_segmentation_schema()`.
    """
    if mode not in GENERATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(GENERATION_MODES.keys())}")

    # Ensure output directory exists and get client
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    # Convert common_prompt to Gemini contents
    system_instruction, contents = common_prompt_to_gemini_contents(common_prompt)

    # Build config
    eff_schema = schema or topic_segmentation_schema()
    final_budget = get_thinking_budget_for(model, thinking_budget)
    base_cfg = dict(**GENERATION_MODES[mode])
    config = _build_config(
        base_cfg,
        final_budget,
        response_mime_type="application/json",
        response_schema=eff_schema,
        system_instruction=system_instruction,
    )

    # Call Gemini API
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        data = json.loads(resp.text)
    except Exception as e:
        print(f"[Error] Model call or JSON parsing failed: {e}")
        return None

    # Save output
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[JSON:{mode}] Saved structured output to {save_path}")
    except Exception as e:
        print(f"[Error] Failed to save JSON output: {e}")
        return None

    return data


# =============================================================================
# BATCH GENERATION FUNCTIONS
# =============================================================================

def batch_generate_text_outputs(
    prompts: List[Dict[str, Any]],
    save_paths: List[str],
    model: str = "gemini-2.0-flash",
    mode: str = "restrictive",
    thinking_budget: Optional[int] = None,
) -> Optional[List[str]]:
    # Basic validations
    if not (isinstance(prompts, list) and all(isinstance(p, dict) for p in prompts)):
        raise ValueError("prompts must be a list of common_prompt dicts")
    if not isinstance(save_paths, list):
        raise ValueError("save_paths must be a list")
    if len(prompts) != len(save_paths):
        raise ValueError("prompts and save_paths must be the same length")

    if mode not in GENERATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(GENERATION_MODES.keys())}")

    # Ensure output directories exist and get client
    for p in save_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    # Convert all common_prompts to Gemini contents
    converted_prompts = []
    for common_prompt in prompts:
        system_instruction, contents = common_prompt_to_gemini_contents(common_prompt)
        converted_prompts.append((system_instruction, contents))

    # --- Build config as a DICTIONARY ---
    final_budget = get_thinking_budget_for(model, thinking_budget)
    config_dict = dict(**GENERATION_MODES[mode])  # Start with base config dict
    
    if final_budget is not None:
        config_dict["thinking_config"] = {"thinking_budget": final_budget}
    # ------------------------------------

    # Build Inline Requests (same model/config for all)
    inline_requests = []
    for system_instruction, contents in converted_prompts:
        request = {
            "contents": contents,
            "generation_config": config_dict
        }
        if system_instruction:
            request["system_instruction"] = system_instruction
        inline_requests.append(request)

    # Call Batch API with client.batches.create()
    print(f"Starting batch job for {len(inline_requests)} prompts...")
    try:
        batch_job = client.batches.create(
            model=model,
            src=inline_requests,
            config={ 'display_name': "batch-text-job" },
        )
        print(f"Batch job created: {batch_job.name}")
        
        # --- 5. Wait for Job to Complete ---
        # This is a simple blocking wait. For production, you'd
        # poll, use callbacks, or handle this asynchronously.
        while batch_job.state.name in ('JOB_STATE_PENDING', 'JOB_STATE_RUNNING'):
            print(f"Waiting for job... (State: {batch_job.state.name})")
            time.sleep(10)  # Poll every 10 seconds
            batch_job = client.batches.get(name=batch_job.name)

        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(f"Batch job failed with state: {batch_job.state.name}")
        
        print("Batch job succeeded.")
        
    except Exception as e:
        print(f"[Error] Batch job failed: {e}")
        return None

    # --- 6. Collect Outputs and Save ---
    # The response is a list of GenerateContentResponse-like dicts
    outputs: list[str] = []
    
    # We must match our saved output paths to the job results.
    # The batch job response preserves the order of the input requests.
    if len(batch_job.dest.inline_responses) != len(save_paths):
         print(f"[Error] Mismatch between request count ({len(save_paths)}) and response count ({len(batch_job.dest.inline_responses)})")
         return None

    for idx, resp_dict in enumerate(batch_job.dest.inline_responses):
        text_out = ""
        try:
            # Convert the dict back to a response object to safely access text
            resp = genai.types.GenerateContentResponse(resp_dict)
            text_out = resp.text or ""
        except Exception as e:
            print(f"[Warn] Could not parse response for item {idx}: {e}")

        outputs.append(text_out)

        try:
            with open(save_paths[idx], "w", encoding="utf-8") as f:
                f.write(text_out)
            print(f"[TextBatch:{mode}] Saved item {idx} to {save_paths[idx]}")
        except Exception as e:
            print(f"[Error] Failed to save item {idx}: {e}")

    return outputs


def batch_generate_structured_outputs(
    prompts: List[Dict[str, Any]],
    save_paths: List[str],
    schema: Optional[genai_types.Schema] = None,
    model: str = "gemini-2.0-flash",
    mode: str = "restrictive",
    thinking_budget: Optional[int] = None,
) -> Optional[List[Dict | List]]:
    """
    Batch-generate structured JSON outputs.
    Each item in `prompts` is a common_prompt dict, and the corresponding result is written to save_paths[idx].
    One schema/model/mode/thinking_budget is applied to the entire batch.
    """
    # Basic validations
    if not (isinstance(prompts, list) and all(isinstance(p, dict) for p in prompts)):
        raise ValueError("prompts must be a list of common_prompt dicts")
    if not isinstance(save_paths, list):
        raise ValueError("save_paths must be a list")
    if len(prompts) != len(save_paths):
        raise ValueError("prompts and save_paths must be the same length")

    if mode not in GENERATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(GENERATION_MODES.keys())}")

    # Ensure output directories exist and get client
    for p in save_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    # Convert all common_prompts to Gemini contents
    converted_prompts = []
    for common_prompt in prompts:
        system_instruction, contents = common_prompt_to_gemini_contents(common_prompt)
        converted_prompts.append((system_instruction, contents))

    # Build config (shared across the batch)
    final_schema = schema or topic_segmentation_schema()
    final_budget = get_thinking_budget_for(model, thinking_budget)
    base_cfg = dict(**GENERATION_MODES[mode])
    
    # Build requests (same model/config for all, but individual system_instructions)
    requests: list = []
    for system_instruction, contents in converted_prompts:
        config = _build_config(
            base_cfg,
            final_budget,
            response_mime_type="application/json",
            response_schema=final_schema,
            system_instruction=system_instruction,
        )
        try:
            req = genai.types.GenerateContentRequest(model=model, contents=contents, config=config)
        except Exception:
            req = {"model": model, "contents": contents, "config": config}
        requests.append(req)

    # Call Gemini batch API
    try:
        batch_resp = client.models.batch_generate_contents(requests=requests)
    except Exception as e:
        print(f"[Error] Batch model call failed: {e}")
        return None

    # Collect parsed JSON and save
    results: list[dict | list] = []
    for idx, resp in enumerate(batch_resp):
        parsed = None
        try:
            text_out = getattr(resp, "text", None)
            if text_out is None and hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                text_out = getattr(cand, "text", "") or getattr(cand, "content", "") or ""
            if text_out is None:
                text_out = ""
            parsed = json.loads(text_out)
        except Exception as e:
            print(f"[Warn] JSON parse failed for item {idx}: {e}")

        results.append(parsed if parsed is not None else [])

        try:
            with open(save_paths[idx], "w", encoding="utf-8") as f:
                if parsed is not None:
                    json.dump(parsed, f, indent=2)
                else:
                    # keep file valid JSON even if parsing failed
                    json.dump([], f, indent=2)
            print(f"[JSONBatch:{mode}] Saved item {idx} to {save_paths[idx]}")
        except Exception as e:
            print(f"[Error] Failed to save item {idx}: {e}")

    return results



# # ---------------------------
# # Example usage
# # ---------------------------
# if __name__ == "__main__":
#     # Example prompt: Dict[str, Any] of strings (could also include image parts)
#     common_prompt = {
#         "blocks": [
#             {"text": "Segment this lecture transcript into sections with index, title, times, and slide ranges."},
#             {"text": "Transcript: <PUT YOUR TEXT HERE>"}
#         ]
#     }
#     # 1. Plain text output with pure deterministic run
#     generate_text_output_gemini(
#         prompt,
#         "output_text.txt",
#         model="gemini-2.0-flash",
#         mode="deterministic"
#     )

#     # 2. Structured JSON output with restrictive flexible run
#     generate_structured_output_gemini(
#         prompt,
#         "output_structured.json",
#         model="gemini-2.5-flash",
#         mode="restrictive",
#         thinking_budget=0
#     )

#     # 3. Batch plain text output (same model/mode/budget for all)
#     prompts = [
#         ["Summarize:", "Doc A ..."],
#         ["Summarize:", "Doc B ..."],
#     ]
#     save_paths = ["out_a.txt", "out_b.txt"]

#     batch_generate_text_outputs(
#         prompts,
#         save_paths,
#         model="gemini-2.5-flash-lite",
#         mode="restrictive"
#     )

#     # 4. Batch structured JSON output — using default schema
#     prompts_structured = [
#         ["Segment this lecture into sections.", "Transcript: <DOC A TEXT>"],
#         ["Segment this lecture into sections.", "Transcript: <DOC B TEXT>"],
#     ]
#     save_paths_structured = ["batch_json/a.json", "batch_json/b.json"]

#     batch_generate_structured_outputs(
#         prompts=prompts_structured,
#         save_paths=save_paths_structured,
#         model="gemini-2.5-flash-lite",   # 2.5 ⇒ default thinking_budget=0
#         mode="restrictive",
#         thinking_budget=None             # leave None to auto-apply defaults
#     )

#     # 5. Batch structured JSON output — with custom schema
#     custom_schema = types.Schema(
#         type="OBJECT",
#         properties={
#             "title": types.Schema(type="STRING"),
#             "bullets": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
#         },
#         required=["title", "bullets"],
#     )

#     prompts_structured2 = [
#         ["Summarize into a title + 3 bullets.", "Content: <DOC C TEXT>"],
#         ["Summarize into a title + 3 bullets.", "Content: <DOC D TEXT>"],
#     ]
#     save_paths_structured2 = ["batch_json_custom/c.json", "batch_json_custom/d.json"]

#     batch_generate_structured_outputs(
#         prompts=prompts_structured2,
#         save_paths=save_paths_structured2,
#         schema=custom_schema,            # custom schema applied to the batch
#         model="gemini-2.5-flash",
#         mode="restrictive",
#         thinking_budget=0                # explicitly set (overrides default)
#     )
