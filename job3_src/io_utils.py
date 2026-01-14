
import json
import os
from typing import Any, Dict, List, Optional



def create_modalities_json_file(dir_path: str, strategy: str = "midpoint", dedupe_adjacent: bool = True, output_filename: str = "extracted_modalities.json") -> None:
    """
    Load ocr_text.json (slides) and transcript_text.json (dict with 'segments') from dir_path,
    merge by time using 'midpoint' (unique) or 'overlap' (multi-assign),
    and save to extracted_modalities.json.
    """
    # ---- load ----
    slides = json.load(open(os.path.join(dir_path, "ocr_text.json"), "r", encoding="utf-8"))["slides"]
    segs   = json.load(open(os.path.join(dir_path, "transcript_text.json"), "r", encoding="utf-8"))["segments"]

    # ---- normalize & sort ----
    slides = [{"index": s["index"], "start": float(s["start"]), "end": float(s["end"]), "filename": s["filename"],
               "ocr_text": s.get("ocr_text",""), "_b": []} for s in slides]
    segs   = [{"start": float(x["start"]), "end": float(x["end"]),
               "text": (x.get("text") or "").strip()} for x in segs if (x.get("text") or "").strip()]

    slides.sort(key=lambda s: s["index"])      # by index only
    segs.sort(key=lambda x: x["start"])        # by start only

    # ---- strategies ----
    if strategy == "overlap":
        def overlaps(a1, a2, b1, b2): 
            return max(a1, b1) < min(a2, b2)

        i = 0
        for s in slides:
            st, en = s["start"], s["end"]
            while i < len(segs) and segs[i]["end"] <= st:
                i += 1
            k = i
            while k < len(segs) and segs[k]["start"] < en:
                if overlaps(st, en, segs[k]["start"], segs[k]["end"]):
                    s["_b"].append(segs[k]["text"])
                k += 1

    elif strategy == "midpoint":
        # precompute midpoints, sort by midpoint
        mids = sorted([((g["start"] + g["end"]) * 0.5, g["text"]) for g in segs], key=lambda t: t[0])
        j = 0
        for s in slides:
            st, en = s["start"], s["end"]
            # advance until mid >= st
            while j < len(mids) and mids[j][0] < st:
                j += 1
            k = j
            while k < len(mids) and mids[k][0] < en:
                s["_b"].append(mids[k][1])
                k += 1
            j = k  # mids are consumed monotonically; no repetition across slides
    else:
        raise ValueError('strategy must be "midpoint" or "overlap"')

    # ---- build & save ----
    out = []
    for s in slides:
        parts = s["_b"]
        if dedupe_adjacent and parts:
            parts = [parts[0]] + [p for p, q in zip(parts[1:], parts[:-1]) if p != q]
        out.append({
            "index": s["index"],
            "start": s["start"],
            "end": s["end"],
            "filename": s.get("filename", ""),
            "transcript_text": " ".join(parts).strip(),
            "ocr_text": s["ocr_text"]
        })

    out_path = os.path.join(dir_path, output_filename)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out_path


def load_segment_modalities(
    dir_path: str,
    modalities_json_filename: str,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load and filter modality segments from JSON file.
    
    Args:
        dir_path: Directory containing the modalities JSON file.
        modalities_json_filename: Name of the JSON file to load.
        start_index: Optional starting slide index (inclusive).
        end_index: Optional ending slide index (inclusive).
        
    Returns:
        List of segment dictionaries sorted by 'index' field.
        Returns empty list if no segments found.
        
    Notes:
        If start_index/end_index not provided, uses first/last segment indices.
    """
    modalities_json_path = os.path.join(dir_path, modalities_json_filename)
    if not os.path.exists(modalities_json_path):
        create_modalities_json_file(dir_path, strategy="midpoint")
    
    with open(modalities_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    segments = sorted(segments, key=lambda s: s.get("index", 0))
    if not segments:
        return []

    start_index = start_index or segments[0]["index"]
    end_index = end_index or segments[-1]["index"]

    return [s for s in segments if start_index <= s.get("index", 0) <= end_index]


def build_transcript_blocks_v1(
    dir_path: str,
    modalities_json_filename: str = "extracted_modalities.json",
) -> Dict[str, Dict[str, Any]]:
    """
    Build transcript-only content blocks organized by slide.
    
    Args:
        dir_path: Directory containing modalities JSON.
        modalities_json_filename: Name of the JSON file to load.
        
    Returns:
        Dictionary mapping segment IDs to content blocks in common_prompt format:
            {
                "segment_id": {
                    "system": None,
                    "blocks": [
                        {"type": "text", "text": "..."},
                        ...
                    ]
                }
            }
    
    Block Structure (per slide):
        Each slide produces ONE text block:
        
            --- Slide {n} ---
            Time (seconds): start={t_start:.2f}, end={t_end:.2f}

            [TRANSCRIPT TEXT]
            {transcript_text}
        
        Final block (text):
            ===== END LECTURE CONTEXT =====
    """

    segment_context = load_segment_modalities(dir_path, modalities_json_filename)

    formatted_blocks: List[Dict[str, Any]] = []
    for slide in segment_context:
        slide_no = slide.get("index")
        t_start = float(slide.get("start", 0.0))
        t_end = float(slide.get("end", 0.0))
        
        transcript_text = (slide.get("transcript_text") or "").strip()
        if not transcript_text:
            transcript_text = "_No transcript text for this slide._"

        text_block = (
            f"--- Slide {slide_no} ---\n"
            f"Time (seconds): start={t_start:.2f}, end={t_end:.2f}\n\n"
            f"[TRANSCRIPT TEXT]\n{transcript_text}\n"
        )
        formatted_blocks.append({"type": "text", "text": text_block})

    # Add global footer as final text block
    formatted_blocks.append({
        "type": "text",
        "text": "===== END LECTURE CONTEXT =====\n"
    })

    model_context = {
        "system": None,
        "blocks": formatted_blocks
    }

    return model_context

