import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import cv2
import json
import time
import pandas as pd
import pytesseract
from pytesseract import Output
import numpy as np
from utils.common import *



# Generic utility functions for OCR processing
def get_sorted_frame_filenames(folder_path):
    """
    Reads a folder and returns a list of frame filenames sorted by timestamp.

    Args:
        folder_path (str): Path to the directory containing frame PNGs.

    Returns:
        list[str]: Sorted list of filenames (not full paths).
    """
    def parse_frame_filename(filename):
        """
        Extract (minutes, seconds, milliseconds) as integers from a filename.
        Example: frame_002.56.000.png -> (2, 56, 0)
        """
        match = re.match(r"frame_(\d{3})\.(\d{2})\.(\d{3})", filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return tuple(map(int, match.groups()))

    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    sorted_files = sorted(all_files, key=parse_frame_filename)
    return sorted_files

def estimate_font_size(median_height: float) -> int:
    """
    Estimate font size from OCR bounding box height.
    Uses an empirically derived linear scaling that works
    across Arial and Calibri test slides.
    """
    return int(median_height * 0.69 + 0.39)

def resize_by_height(img, target_height=1080, interpolation=cv2.INTER_CUBIC):
    """
    Resize image to a fixed target height (1080px) while keeping aspect ratio.
    """
    h, w = img.shape[:2]
    scale = target_height / float(h)
    target_width = int(w * scale)
    resized = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
    return resized

def run_ocr(image_path: str, psm: int = 4, debug: bool = False):
    """
    Run OCR with configurable Tesseract Page Segmentation Mode (PSM).

    Args:
        image_path (str): Path to the input image.
        psm (int, optional): Page Segmentation Mode.
            1 = Automatic page segmentation with OSD.
            3 = Fully automatic page segmentation (default).
            4 = Assume a single column of text (good for slides/papers).
            11 = Sparse text (find text anywhere, no ordering).
        debug (bool, optional): If True, prints config and a preview of OCR output.

    Returns:
        pd.DataFrame: Cleaned OCR results (safe text + numeric confidence).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # --- Resize to standard height (maintain aspect ratio) ---
    img = resize_by_height(img, interpolation=cv2.INTER_CUBIC)

    # --- OCR ---
    custom_config = f"--psm {psm}"
    df = pytesseract.image_to_data(img, output_type=Output.DATAFRAME, config=custom_config)

    # --- Clean dataframe ---
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(int)

    if debug:
        print(f"Running OCR with config: {custom_config}")
        print(df.to_string(index=False))

    return df


# Visualization functions
def draw_ocr_boxes(image_path: str, psm: int = 4, 
                   draw_blocks: bool = False, 
                   draw_paragraphs: bool = False, 
                   draw_lines: bool = False, 
                   conf_threshold: int = 40, 
                   debug: bool = False, 
                   output_path: str = "experiments/output/ocr_boxes.png"):
    """
    Draw OCR bounding boxes (block / paragraph / line) on an image.

    Args:
        image_path (str): Path to input image.
        psm (int): Tesseract Page Segmentation Mode.
        draw_blocks (bool): Draw block-level boxes.
        draw_paragraphs (bool): Draw paragraph-level boxes.
        draw_lines (bool): Draw line-level boxes.
        conf_threshold (int): Confidence threshold for filtering words.
        debug (bool): Print debug info.
        output_path (str): File path to save annotated image.
    """
    df = run_ocr(image_path, psm=psm, debug=False)
    df = df[(df["text"].str.strip() != "") & (df["conf"] >= conf_threshold)]

    img = cv2.imread(image_path)
    img = resize_by_height(img)  # keep consistent with OCR

    # --- Block-level ---
    if draw_blocks:
        for block_num, block_df in df.groupby("block_num"):
            x1, y1 = block_df[["left", "top"]].min()
            x2 = (block_df["left"] + block_df["width"]).max()
            y2 = (block_df["top"] + block_df["height"]).max()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
            if debug:
                text = " ".join(block_df["text"].tolist())
                print(f"[Block {block_num}] bbox=({x1},{y1},{x2},{y2}) | Text: {text}")

    # --- Paragraph-level ---
    if draw_paragraphs:
        for (block, par), par_df in df.groupby(["block_num", "par_num"]):
            x1, y1 = par_df[["left", "top"]].min()
            x2 = (par_df["left"] + par_df["width"]).max()
            y2 = (par_df["top"] + par_df["height"]).max()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            if debug:
                text = " ".join(par_df["text"].tolist())
                print(f"[Paragraph {block},{par}] bbox=({x1},{y1},{x2},{y2}) | Text: {text}")

    # --- Line-level ---
    if draw_lines:
        for (block, par, line), line_df in df.groupby(["block_num", "par_num", "line_num"]):
            x1, y1 = line_df[["left", "top"]].min()
            x2 = (line_df["left"] + line_df["width"]).max()
            y2 = (line_df["top"] + line_df["height"]).max()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green
            if debug:
                text = " ".join(line_df["text"].tolist())
                print(f"[Line {block},{par},{line}] bbox=({x1},{y1},{x2},{y2}) | Text: {text}")

    # Save image
    cv2.imwrite(output_path, img)
    print(f"✅ Annotated image saved to {output_path}")
    return output_path



# OCR text extraction functions
def extract_ocr_plain_text(image_path: str, psm: int = 4, debug: bool = False) -> str:
    """
    Extract plain text from a slide image using OCR.
    Resizes image to fixed height (1080px) while maintaining aspect ratio.

    Args:
        image_path (str): Path to the input slide image.
        psm (int, optional): Tesseract Page Segmentation Mode.
            1 = Automatic page segmentation with OSD.
            3 = Fully automatic page segmentation (default).
            4 = Assume a single column of text (good for slides/papers).
            11 = Sparse text (find text anywhere, no ordering).

    Returns:
        str: Extracted plain text.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to fixed height = 1080px
    img = resize_by_height(img)

    # OCR
    custom_config = f"--psm {psm}"
    text = pytesseract.image_to_string(img, config=custom_config)
    if debug:
        print("Extracted plain text:::")
        print(text.strip())
        print("---------------------------")

    return text.strip()

def extract_ocr_structured_text(image_path: str, psm: int = 4, font_size_method: str = "word_height", debug: bool = False):
    """
    Run OCR and save results into a plain text file.
    Font size tag is left-aligned, then indentation spaces, then the text.

    Args:
        image_path (str): Path to input image.
        psm (int, optional): Tesseract Page Segmentation Mode.
        debug (bool, optional): Print debug information.
    """
    # Run OCR with word-level info
    df = run_ocr(image_path, psm=psm, debug=False)
    clean_df = df[df["text"] != ""]

    # --- Character-level OCR if needed ---
    char_boxes = None
    if font_size_method == "char_height":
        img = cv2.imread(image_path)
        img = resize_by_height(img)
        h, w = img.shape[:2]
        boxes = pytesseract.image_to_boxes(img, config=f"--psm {psm}")
        char_boxes = []
        for b in boxes.splitlines():
            b = b.split()
            if len(b) == 6:
                _, x1, y1, x2, y2, _ = b
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # flip y because tesseract origin is bottom-left
                y1, y2 = h - y1, h - y2
                char_boxes.append((x1, y2, x2, y1))  # (x1,y1,x2,y2)

    # --- Prepare output lines ---
    lines = []
    for (page, block, par), par_df in clean_df.groupby(["page_num", "block_num", "par_num"]):
        block_left = par_df["left"].min()  # reference left for this paragraph
        #par_font_size = int(par_df["height"].median())  # consistent font size for the whole paragraph

        for (_, _, _, line_num), line_df in par_df.groupby(["page_num", "block_num", "par_num", "line_num"]):
            line_words = line_df.sort_values("left")  # ensure left-to-right order

            # Estimate font size
            font_size = estimate_font_size(line_words["height"].median())
            if font_size_method == "char_height" and char_boxes:
                # get chars inside line box
                x1, y1 = int(line_df["left"].min()), int(line_df["top"].min())
                x2, y2 = int(line_df["left"].max() + line_df["width"].max()), int(line_df["top"].max() + line_df["height"].max())
                line_char_heights = [cb[3] - cb[1] for cb in char_boxes if cb[0] >= x1 and cb[2] <= x2 and cb[1] >= y1 and cb[3] <= y2]
                font_size = estimate_font_size(np.median(line_char_heights)) if line_char_heights else font_size               
            
            # Estimate indentation
            indent_spaces = int((line_df["left"].min() - block_left) / 20)

            # Preserve gaps between words
            line_text_parts = []
            prev_right = None

            for _, word in line_words.iterrows():
                if prev_right is not None:
                    # gap = current word's left - previous word's right
                    gap = word["left"] - prev_right
                    gap = estimate_font_size(gap)  # convert gap to spaces
                    spaces = int(gap / 20)  
                    line_text_parts.append(" " * min(max(spaces, 1), 4))  # limit to max 4 spaces

                line_text_parts.append(word["text"])
                prev_right = word["left"] + word["width"]

            line_text = "".join(line_text_parts)

            lines.append(f"[font_size={int(font_size)}] " + " " * indent_spaces + line_text)
        # --- empty line after paragraph ---
        lines.append("")

    text = "\n".join(lines).strip()
    if debug:
        print("Extracted structured text:::")
        print(text)
        print("-----------------------------")
    return text


def extract_slide_ocr_text(video_path, slides_dir="slides", psm=4, output_file=None, debug=False):
    """
    Extract OCR text from slide images and save structured JSON ocr text.

    Parameters:
        video_path (str): Path to the video file (used to build slides dir path).
        slides_dir (str): Relative folder name where slide images are stored.
        psm (int): Tesseract Page Segmentation Mode.
        output_dir (str): Directory to save OCR text JSON.
        debug (bool): Print debugging info.
    """
    # Construct slide folder path
    slide_folder = os.path.join(os.path.dirname(video_path), slides_dir)

    if not os.path.exists(slide_folder):
        raise FileNotFoundError(f"Slides folder not found: {slide_folder}")

    # Gather slide images (sorted)
    slide_files = get_sorted_frame_filenames(slide_folder)
    if not slide_files:
        raise FileNotFoundError(f"No slide images found in: {slide_folder}")

    slides = []

    def parse_timestamp(fname):
        name = os.path.basename(fname).replace("frame_", "").replace(".png", "")
        minutes, seconds, millis = map(int, name.split("."))
        return minutes * 60 + seconds + millis / 1000.0

    for idx, fname in enumerate(slide_files):
        img_path = os.path.join(slide_folder, fname)
        end_time = parse_timestamp(fname)
        start_time = 0.0 if idx == 0 else parse_timestamp(slide_files[idx - 1])

        ocr_text = extract_ocr_plain_text(img_path, psm=psm, debug=debug)

        slides.append({
            "index": idx+1,
            "start": start_time,
            "end": end_time,
            "filename": fname,
            "ocr_text": ocr_text
        })

    ocr_text = {"slides": slides}

    # Save JSON
    if output_file:
        json_save_path = os.path.join(os.path.dirname(video_path), output_file)
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(ocr_text, f, ensure_ascii=False, indent=2)

    print(f"OCR slide ocr_text saved to: {json_save_path}")


if __name__ == "__main__":
    helper = Helper()
    start_time = time.time()

    # Configure file and directory paths
    video_path = "output/web_vpp/v8020/BIOL2301_lecture05_0609_8020.mp4"



    extract_slide_ocr_text(video_path, output_file="ocr_text.json")
 

    print(f"\n✅ OCR processing completed in {time.time() - start_time:.2f} seconds.")
