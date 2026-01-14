import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import shutil
import time
import math
import matplotlib.pyplot as plt
from collections import deque

from utils.common import *


# --- Utility Functions ---

def read_frame(cap, frame_index):
    """
    Read a frame from the video at a given index, convert it to grayscale,
    and generate a timestamp-based filename.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        frame_index (int): Index of the frame to read.

    Returns:
        tuple: (frame_bgr, frame_gray, frame_filename) if successful, else None.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_bgr = cap.read()
    if not ret:
        return None

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Compute timestamp-based filename
    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    minutes = int(timestamp_ms // 60000)
    seconds = int((timestamp_ms % 60000) // 1000)
    millis = int(timestamp_ms % 1000)
    frame_filename = f"frame_{minutes:03d}.{seconds:02d}.{millis:03d}.png"

    return frame_bgr, frame_gray, frame_filename

def visualize_images(image_dict, title="Visualization of Images"):
    """
    Plots images from a dictionary in a grid layout.
    The grid shape is automatically chosen based on the number of images.
    
    Args:
        image_dict (dict): A dictionary with string keys and image arrays.
        title (str): Optional figure title.
    """
    num_images = len(image_dict)
    if num_images == 0:
        print("No images to display.")
        return
    
    # Determine grid size (try to keep it square)
    cols = 2 if num_images <= 4 else 3
    rows = int(math.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    
    # Flatten axes for easy iteration, even if it's 1D
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (key, img_array) in enumerate(image_dict.items()):
        if img_array is None:  # skip missing images
            continue
        ax = axes[i]
        cmap = 'gray' if len(img_array.shape) == 2 else None
        ax.imshow(img_array, cmap=cmap)
        ax.set_title(key)
        # ax.axis("off")
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    #plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def copy_all_files(src_folder: str, dst_folder: str, clean_output=True):
    """
    Copy all files from src_folder to dst_folder.

    Args:
        src_folder (str): Path to the source folder.
        dst_folder (str): Path to the destination folder.
    """
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"Source folder not found: {src_folder}")

    if clean_output and os.path.exists(dst_folder): shutil.rmtree(dst_folder)
    os.makedirs(dst_folder, exist_ok=True)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        if os.path.isfile(src_path):  # skip subfolders
            shutil.copy(src_path, dst_path)

    print(f"✅ All files copied from {src_folder} to {dst_folder}")



# --- Processing Helpers ---

def get_speaker_mask(
    frame_window,
    motion_std_threshold=15,
    diff_threshold=20,
    persistence_fraction=1/3,
    min_active_frames=3,
    closing_kernel_size=(25, 25),
    dilation_kernel_size=(5, 5),
    dilation_iterations=2
    ):
    """
    Analyzes a window of frames to create a mask of moving regions (like a speaker)
    based on motion characteristics.

    The logic combines two main ideas:
    1. High Variance: Speaker regions have high pixel value variance over time due to constant movement.
    2. Motion Persistence: Speaker movement is persistent, unlike slide animations which happen once.

    Args:
        frame_window (list): List of consecutive grayscale frames.
        motion_std_threshold (float): Std deviation threshold to identify high-variance pixels.
        diff_threshold (int): Threshold for binarizing per-pixel frame differences to detect any motion.
        persistence_fraction (float): Fraction of frames in the window where motion must persist for a pixel to be considered.
        min_active_frames (int): An absolute minimum number of frames a pixel must be active in.
        closing_kernel_size (tuple): Kernel size for morphological closing to fill holes in the mask.
        dilation_kernel_size (tuple): Kernel size for morphological dilation to expand the mask.
        dilation_iterations (int): Number of times to apply dilation.

    Returns:
        numpy.ndarray: The final binary speaker mask.
    """
    if not frame_window or len(frame_window) < 2:
        return None

    # 1. Calculate temporal standard deviation to find high-variance regions
    frame_stack = np.stack(frame_window, axis=2)
    std_dev_map = np.std(frame_stack, axis=2)

    # 2. Calculate motion persistence to distinguish from non-animated content
    diffs = [cv2.absdiff(frame_window[i], frame_window[i+1]) for i in range(len(frame_window)-1)]
    diffs_bin = [(d > diff_threshold).astype(np.uint8) for d in diffs]
    activity_count = np.sum(diffs_bin, axis=0)

    # A pixel is persistent if it's active for a certain fraction of the window
    required_active_frames = max(min_active_frames, int(len(frame_window) * persistence_fraction))
    persistence_mask = (activity_count >= required_active_frames).astype(np.uint8) * 255

    # 3. Combine variance and persistence to create the initial raw mask
    raw_mask = ((std_dev_map > motion_std_threshold) & (persistence_mask > 0)).astype(np.uint8) * 255

    # 4. Use morphological closing to fill small holes and consolidate regions
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, closing_kernel)

    # 5. Use morphological dilation to expand the mask slightly to ensure full coverage
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    final_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=dilation_iterations)

    return final_mask

def step1_compare_frames(current_gray, next_gray, metric='mse', mask=None):
    """
    Step 1 frame comparison for initial filtering.

    Compares two frames using a specified metric (MSE, SSIM).
    Optionally applies a speaker mask to exclude regions.

    Args:
        current_gray (numpy.ndarray): Current grayscale frame.
        next_gray (numpy.ndarray): Next grayscale frame.
        metric (str): Comparison metric: 'mse', 'ssim'.
        mask (numpy.ndarray, optional): A mask to exclude regions from comparison.

    Returns:
        float: Similarity/difference score depending on metric.
    """
    # Apply mask if provided
    if mask is not None:
        inverted_mask = cv2.bitwise_not(mask)
        current_gray = cv2.bitwise_and(current_gray, current_gray, mask=inverted_mask)
        next_gray = cv2.bitwise_and(next_gray, next_gray, mask=inverted_mask)

    if metric == 'mse':
        # Mean Squared Error
        err = np.sum((current_gray.astype("float") - next_gray.astype("float")) ** 2)
        err /= float(current_gray.shape[0] * current_gray.shape[1])
        return err

    elif metric == 'ssim':
        # Structural Similarity Index (higher = more similar)
        score, _ = ssim(current_gray, next_gray, full=True)
        return score

    else:
        raise ValueError(f"Unknown metric: {metric}")
    

def canny_edges(
    gray_image,
    blur_kernel_size=(5, 5),
    canny_threshold1=50,
    canny_threshold2=150,
    dilation_kernel_size=(5, 5),
    dilation_iterations=1
):
    """
    Applies blurring, Canny edge detection, and dilation to a grayscale image.

    Args:
        gray_image (np.ndarray): Input image in grayscale format.
        blur_kernel_size (tuple): Kernel size for Gaussian blur (default: (5, 5)).
        canny_threshold1 (int): Lower threshold for Canny edge detection (default: 50).
        canny_threshold2 (int): Upper threshold for Canny edge detection (default: 150).
        dilation_kernel_size (tuple): Kernel size for dilation (default: (5, 5)).
        dilation_iterations (int): Number of dilation iterations (default: 1).

    Returns:
        np.ndarray: Binary edge mask after Canny detection and dilation.
    """
    # Step 1: Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray_image, blur_kernel_size, 0)

    # Step 2: Canny edge detection
    edges_mask = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Step 3: Dilation to connect broken edges
    kernel = np.ones(dilation_kernel_size, np.uint8)
    dilated_edges_mask = cv2.dilate(edges_mask, kernel, iterations=dilation_iterations)

    return dilated_edges_mask

def detect_optical_flow_changes_with_frame_diff(frame1_gray, frame2_gray, diff_edges_current, flow_thresh=2.0, diff_thresh=25, blur_kernel=3):
    """
    Detect local regions of change using optical flow magnitude,
    constrained by regions where the absolute frame difference indicates change.

    Args:
        frame1_gray (np.ndarray): First grayscale frame (uint8).
        frame2_gray (np.ndarray): Second grayscale frame (uint8).
        flow_thresh (float): Threshold for optical flow magnitude.
        diff_thresh (int): Threshold for absolute difference between frames (0-255).
        blur_kernel (int): Gaussian blur kernel size for magnitude smoothing.

    Returns:
        change_mask (np.ndarray): Binary mask (0/255) of detected changes.
    """
    frame1_gray = frame1_gray.astype(np.uint8)
    frame2_gray = frame2_gray.astype(np.uint8)

    # --- Dense optical flow ---
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    # --- Absolute frame difference ---
    frame_diff = cv2.convertScaleAbs(diff_edges_current)
    _, frame_diff_mask = cv2.threshold(frame_diff, diff_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    frame_diff_mask = cv2.dilate(frame_diff_mask, kernel, iterations=1)

    # --- Blur magnitude to reduce noise ---
    magnitude_blur = cv2.GaussianBlur(magnitude, (blur_kernel, blur_kernel), 0)

    # --- Threshold optical flow magnitude ---
    flow_image_denoised = (magnitude_blur > flow_thresh).astype(np.uint8) * 255

    # --- Combine flow mask with absolute frame difference ---
    flow_masked_on_diff = cv2.bitwise_and(flow_image_denoised, frame_diff_mask)
    percentage_flowed = round(100* cv2.countNonZero(flow_masked_on_diff)/cv2.countNonZero(frame_diff_mask), 1)

    # --- Morphological cleanup ---
    kernel = np.ones((3,3), np.uint8)
    flow_masked_on_diff = cv2.morphologyEx(flow_masked_on_diff, cv2.MORPH_OPEN, kernel, iterations=1)
    flow_masked_on_diff = cv2.dilate(flow_masked_on_diff, kernel, iterations=1)

    return flow_image_denoised, flow_masked_on_diff, percentage_flowed

def step2_compare_frames(
    current_frame: np.ndarray,
    next_frame: np.ndarray,
    keep_mask: np.ndarray | None = None,
    uniqueness_threshold: float = 50.0,
    flow_threshold: float = 50.0
) -> tuple[bool, dict]:
    """
    Compare two consecutive video frames to decide if the next frame is a new slide.

    Uses:
      - Edge-based masked differences
      - Optional speaker masks to ignore presenter
      - Optical flow to filter gradual/animated changes

    Args:
        current_frame, next_frame: Input BGR frames.
        keep_mask: Optional speaker mask.
        uniqueness_threshold: MSE threshold to trigger flow check.
        flow_threshold: Flow percentage threshold for valid transition.

    Returns:
        duplicate_flag: True if frames are duplicates, False if new slide.
        image_dict: Intermediate visualization images (gray frames, diffs, flow).
    """
    mse_current, mse_next = -1.0, -1.0
    # Convert to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Frame-to-frame absolute difference
    frame_diff = cv2.absdiff(current_gray, next_gray).astype("float")

    # Step 2: Edge detection on both frames
    current_edges = canny_edges(current_gray, blur_kernel_size=(5, 5), dilation_kernel_size=(5, 5))
    # next_edges = canny_edges(next_gray, blur_kernel_size=(5, 5), dilation_kernel_size=(5, 5))

    # Step 3: Restrict differences to edge regions
    diff_current = cv2.bitwise_and(frame_diff, frame_diff, mask=current_edges)
    # diff_next = cv2.bitwise_and(frame_diff, frame_diff, mask=next_edges)

    # Step 4: Apply speaker masks (if provided)
    if keep_mask is not None:
        diff_current = cv2.bitwise_and(diff_current, diff_current, mask=keep_mask)
        # diff_next = cv2.bitwise_and(diff_next, diff_next, mask=keep_mask)

    # Mean squared differences
    mse_current = np.sum(diff_current ** 2) / frame_diff.size
    # mse_next = np.sum(diff_next ** 2) / frame_diff.size

    # Optical flow check (only if edge-diff is high enough)
    flow_image, flow_mask, percent_flow = None, None, None
    duplicate_flag = True
    if mse_current > uniqueness_threshold:
        flow_image, flow_mask, percent_flow = detect_optical_flow_changes_with_frame_diff(
            current_gray, next_gray, diff_current,
            flow_thresh=2.0, diff_thresh=25
        )
        duplicate_flag = percent_flow < flow_threshold

    print(f"{mse_current:4.0f} ({percent_flow}) > {mse_next:4.0f}", end="")

    # Collect visualizations
    image_dict = {
        "Current Frame": current_gray,
        "Next Frame": next_gray,
        "Frame Diff": frame_diff,
        "Diff on Current Edges": diff_current,
        "Optical Flow Image": flow_image,
        "Optical Flow Mask": flow_mask,
    }

    return duplicate_flag, image_dict

def build_joint_mask_window(manual_files, auto_files, m_idx, a_idx, slides_manual, slides_auto, half_window=6):
    """
    Build a frame window combining manual and auto frames around current indices.

    Args:
        manual_files (list[str]): Sorted list of manual frame filenames.
        auto_files (list[str]): Sorted list of auto frame filenames.
        m_idx (int): Current manual index.
        a_idx (int): Current auto index.
        slides_manual (str): Path to manual frames folder.
        slides_auto (str): Path to auto frames folder.
        half_window (int): Number of frames to include from each side (default 6).

    Returns:
        list[np.ndarray]: Grayscale frames for mask computation.
    """
    frames = []

    # Collect manual window
    for j in range(max(0, m_idx - half_window//2), min(len(manual_files), m_idx + half_window//2)):
        img = cv2.imread(os.path.join(slides_manual, manual_files[j]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    # Collect auto window
    for j in range(max(0, a_idx - half_window//2), min(len(auto_files), a_idx + half_window//2)):
        img = cv2.imread(os.path.join(slides_auto, auto_files[j]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    return frames



# --- Main Pipeline Steps ---

def step1_basic_filter(
    video_path, 
    root_output_dir=None, 
    processing_fps=2, 
    speaker_mask_window_size=11, 
    enable_masking=True, 
    mse_threshold=50, 
    manual_slide_bbox=None,
    skip_on_existing=False,
):
    """
    Step 1: Quickly filter out identical or near-identical consecutive frames using MSE.

    Frames are sampled at the target FPS and compared with MSE. Optionally, 
    a speaker mask is computed over a sliding window to reduce noise from 
    presenter motion. If provided, a manual slide ROI limits filtering to 
    that region. Kept frames and masks are stored in 'temp_step1_candidates' 
    and 'temp_step1_masks'.

    Args:
        video_path (str): Path to the video file.
        root_output_dir (str): Root directory for outputs (default = video_folder).
        processing_fps (int): Frame sampling rate.
        speaker_mask_window_size (int): Window size (odd) for mask estimation.
        enable_masking (bool): Whether to apply speaker masking.
        mse_threshold (int): MSE threshold for frame uniqueness.
        manual_slide_bbox (tuple[int,int,int,int] | None): Median Bounding Box (x, y, w, h) defining the slide area.
        skip_on_existing (bool): Skip if outputs already exist.
    """
    start_time = time.time()
    
    if root_output_dir is None:
        root_output_dir = os.path.dirname(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    stride = 15 if video_fps == 0 else max(1, round(video_fps / processing_fps))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define input and output folders
    step1_candidates_folder = os.path.join(root_output_dir, "temp", "temp_step1_candidates")
    step1_masks_folder = os.path.join(root_output_dir, "temp", "temp_step1_masks")

    if skip_on_existing and os.path.exists(step1_candidates_folder) and os.path.exists(step1_masks_folder):
        print(f"Step 1 output already exists. Skipping processing for {root_output_dir}.")
        return

    # Clean and recreate output folders
    if os.path.exists(step1_candidates_folder): shutil.rmtree(step1_candidates_folder)
    if os.path.exists(step1_masks_folder): shutil.rmtree(step1_masks_folder)
    os.makedirs(step1_candidates_folder, exist_ok=True)
    os.makedirs(step1_masks_folder, exist_ok=True)

    print(f"\n--- Starting Step 1: Initial Frame Filtering ---")
    print(f"Target FPS: {processing_fps}, Stride: {stride}, Speaker Masking: {'On' if enable_masking else 'Off'}")

    # Precompute indices of frames we will process
    frame_indices = list(range(0, frame_count, stride))

    # Initialize frame buffer
    frame_buffer = deque()
    half = speaker_mask_window_size // 2
    speaker_mask_window_size = 2 * half + 1  # ensure odd
    for preload_idx in frame_indices[:half + 1]:
        frame_info = read_frame(cap, preload_idx)
        if frame_info:
            frame_buffer.append(frame_info)
    buffer_end_index = half


    candidate_count = 0
    speaker_mask = None
    diff_scores = {}

    # Main loop
    for idx in range(len(frame_indices)-1):
        # Pick the center frame from buffer
        current_frame_idx = min(half, buffer_end_index - half)
        current_frame_bgr, current_frame_gray, current_frame_timestamp = frame_buffer[current_frame_idx]
        _, next_frame_gray, _ = frame_buffer[current_frame_idx+1]

        # Update buffer: add next frame until we reach the end, drop oldest if needed
        if buffer_end_index + 1 < len(frame_indices):
            next_frame_info = read_frame(cap, frame_indices[buffer_end_index + 1])
            if next_frame_info:
                frame_buffer.append(next_frame_info)
                buffer_end_index += 1
        if len(frame_buffer) > speaker_mask_window_size:
            frame_buffer.popleft()

        # Update speaker mask periodically
        if enable_masking:
            frames_window = [gray for _, gray, _ in frame_buffer]
            if idx % (speaker_mask_window_size // 3) == 0:
                speaker_mask = get_speaker_mask(frames_window)

                if manual_slide_bbox:
                    (x, y, w, h) = manual_slide_bbox
                    slide_mask_inv = np.ones(speaker_mask.shape, dtype=np.uint8) * 255
                    cv2.rectangle(slide_mask_inv, (x, y), (x + w, y + h), 0, -1)
                    speaker_mask = cv2.bitwise_or(speaker_mask, slide_mask_inv)
                    
        # Compare current frame with last kept
        score = step1_compare_frames(current_frame_gray, next_frame_gray, metric='mse', mask=speaker_mask)
        diff_scores[current_frame_timestamp] = score

        if score > mse_threshold:
            cv2.imwrite(os.path.join(step1_candidates_folder, current_frame_timestamp), current_frame_bgr)
            if speaker_mask is not None:
                cv2.imwrite(os.path.join(step1_masks_folder, current_frame_timestamp), speaker_mask)
            candidate_count += 1

        # Progress display
        progress = (idx + 1) / len(frame_indices)
        print(
            f"\rProcessing frame {idx+1}/{len(frame_indices)} "
            f"[{int(progress*20)*'='}>{(19-int(progress*20))*' '}] "
            f"{int(progress*100)}%", end=""
        )

    last_frame_bgr, _, last_frame_timestamp = frame_buffer[-1]
    cv2.imwrite(os.path.join(step1_candidates_folder, last_frame_timestamp), last_frame_bgr)
    if speaker_mask is not None:
        cv2.imwrite(os.path.join(step1_masks_folder, last_frame_timestamp), speaker_mask)
    candidate_count += 1

    cap.release()
    print(f"\nStep 1 complete. Found {candidate_count} candidate frames.")

    helper.write_to_json(diff_scores, os.path.join(step1_candidates_folder, "mse_diff_scores.json"))
    print(f"⏱️  step1_basic_filter completed in {time.time() - start_time:.2f}s")

def step2_transition_filter(
    video_path: str,
    root_output_dir=None,
    uniqueness_threshold=50,
    manual_slide_bbox=None,
    skip_on_existing=False,
):
    """
    Step 2: Remove intermediate frames caused by slide transitions.

  
    This step removes redundant frames created by within-slide
    transitions (e.g., animations, gradual content reveals), 
    keeping only the final, fully updated slide states. 
    Each candidate is compared with its predecessor using edge-based masked 
    differences. Speaker masks from Step 1 (or a manual bounding box) help 
    ignore presenter regions. The last valid frame is always preserved until 
    new content appears. Outputs are saved in 'temp_step2_candidates' and 
    'temp_step2_masks'.

    Args:
        video_path (str): Path to the video file.
        root_output_dir (str): Root directory for outputs (default = directory of video_path).
        uniqueness_threshold (int): Minimum edge-diff score for uniqueness.
        manual_slide_bbox (tuple[int,int,int,int] | None): Median Bounding Box (x, y, w, h) defining the slide area.
        skip_on_existing (bool): Skip if outputs already exist.
    """
    start_time = time.time()
    
    if root_output_dir is None:
        root_output_dir = os.path.dirname(video_path)

    # Define input and output folders
    step1_candidates_folder = os.path.join(root_output_dir, "temp", "temp_step1_candidates")
    step1_masks_folder = os.path.join(root_output_dir, "temp", "temp_step1_masks")
    step2_candidates_folder = os.path.join(root_output_dir, "temp", "temp_step2_candidates")
    step2_masks_folder = os.path.join(root_output_dir, "temp", "temp_step2_masks")

    if skip_on_existing and os.path.exists(step2_candidates_folder) and os.path.exists(step2_masks_folder):
        print(f"Step 2 output already exists. Skipping processing for {root_output_dir}.")
        return
    
    # Clean and recreate output folders
    if os.path.exists(step2_candidates_folder): shutil.rmtree(step2_candidates_folder)
    if os.path.exists(step2_masks_folder): shutil.rmtree(step2_masks_folder)
    os.makedirs(step2_candidates_folder, exist_ok=True)
    os.makedirs(step2_masks_folder, exist_ok=True)

    # Collect and sort candidate frames
    sorted_files = helper.get_sorted_frame_filenames(step1_candidates_folder)

    count = 0

    for i in range(len(sorted_files)-1):
        current_frame_filename = sorted_files[i]
        next_frame_filename = sorted_files[i+1]

        current_frame_path = os.path.join(step1_candidates_folder, current_frame_filename)
        current_mask_path = os.path.join(step1_masks_folder, current_frame_filename)

        next_frame_path = os.path.join(step1_candidates_folder, next_frame_filename)
        next_mask_path = os.path.join(step1_masks_folder, next_frame_filename)

        # Load frames
        current_frame_bgr = cv2.imread(current_frame_path)
        next_frame_bgr = cv2.imread(next_frame_path)

        current_frame_mask = cv2.imread(current_mask_path, cv2.IMREAD_GRAYSCALE)
        next_frame_mask = cv2.imread(next_mask_path, cv2.IMREAD_GRAYSCALE)

        if current_frame_mask is not None or next_frame_mask is not None:
            _, mask_cur = cv2.threshold(current_frame_mask, 127, 255, cv2.THRESH_BINARY)
            _, mask_nxt = cv2.threshold(next_frame_mask, 127, 255, cv2.THRESH_BINARY)
            union_mask = cv2.bitwise_or(mask_cur, mask_nxt)
            keep_mask = cv2.bitwise_not(union_mask)
            
        if manual_slide_bbox:
            (x, y, w, h) = manual_slide_bbox
            slide_mask = np.zeros(current_frame_mask.shape, dtype=np.uint8)
            cv2.rectangle(slide_mask, (x, y), (x + w, y + h), 255, -1)
            keep_mask = cv2.bitwise_and(keep_mask, slide_mask)

        # Compare frames using step2 logic
        duplicate_frame, image_dict = step2_compare_frames(current_frame_bgr, next_frame_bgr, keep_mask, uniqueness_threshold)
        
        if not duplicate_frame:
            cv2.imwrite(os.path.join(step2_candidates_folder, current_frame_filename), current_frame_bgr)
            cv2.imwrite(os.path.join(step2_masks_folder, current_frame_filename), cv2.bitwise_not(keep_mask))
            print(f" -----------------> saved ({os.path.basename(current_frame_filename)})")
            count += 1

        else:
            print(f" - ({os.path.basename(current_frame_filename)[6:-4]})")

        # visualize_images(image_dict, title="Step 2 Comparison Visualization")
        # if i == 2: break

    cv2.imwrite(os.path.join(step2_candidates_folder, next_frame_filename), next_frame_bgr)
    cv2.imwrite(os.path.join(step2_masks_folder, next_frame_filename), cv2.bitwise_not(keep_mask))
    print(f"Step 2 complete. Kept {count+1} frames out of {len(sorted_files)}.")

    print("Copying files to final destination folder...")
    final_dest_folder = os.path.join(root_output_dir, "slides")
    copy_all_files(step2_candidates_folder, final_dest_folder)
    print(f"⏱️  step2_transition_filter completed in {time.time() - start_time:.2f}s")

def step3_filter_with_sliding_window(
    video_path: str,
    root_output_dir: str = None,
    window_size: int = 10,
    mse_threshold: int = 50,
    skip_on_existing: bool = False,
):
    """
    Step 3: Deduplicate non-consecutive repeats with a sliding window.

    Frames from Step 2 are compared against a buffer of recent unique frames 
    using MSE to catch revisited slides. Unique frames are saved in 
    'temp_step3_candidates' and copied to 'slides' for final use.

    Args:
        video_path (str): Path to the video file.
        root_output_dir (str): Root directory for outputs (default = directory of video_path).
        window_size (int): Number of past frames to compare against.
        mse_threshold (int): MSE threshold for uniqueness.
        skip_on_existing (bool): Skip if outputs already exist.

    Returns:
        list[str]: Filenames of final unique frames.
    """
    start_time = time.time()
    
    if root_output_dir is None:
        root_output_dir = os.path.dirname(video_path)

    # 1. Define input and output folders
    input_image_folder = os.path.join(root_output_dir, "temp", "temp_step2_candidates")
    input_mask_folder = os.path.join(root_output_dir, "temp", "temp_step2_masks")
    temp_step3_candidates = os.path.join(root_output_dir, "temp", "temp_step3_candidates")

    if skip_on_existing and os.path.exists(temp_step3_candidates):
        print(f"Step 3 output already exists. Skipping processing for {root_output_dir}.")
        return
    
    # Clean and recreate output folders
    if os.path.exists(temp_step3_candidates): shutil.rmtree(temp_step3_candidates)
    os.makedirs(temp_step3_candidates, exist_ok=True)

    print("\n--- Starting Step 3: Non-Consecutive Deduplication ---")
    print(f"Window Size: {window_size}, MSE Threshold: {mse_threshold}")

    # 2. Get and chronologically sort frames from the previous step
    candidate_filenames = helper.get_sorted_frame_filenames(input_image_folder)
    if not candidate_filenames:
        print("No candidate frames to process from Step 2.")
        return

    # 3. Initialize data structures for the sliding window logic
    unique_filenames = []
    # A deque with a max length is an efficient sliding window
    comparison_window_hashes = deque(maxlen=window_size)

    # 4. Iterate through each candidate frame
    for idx, filename in enumerate(candidate_filenames):
        frame_path = os.path.join(input_image_folder, filename)
        mask_path = os.path.join(input_mask_folder, filename)

        frame_gray = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
        frame_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if frame_mask is not None:
            _, mask = cv2.threshold(frame_mask, 127, 255, cv2.THRESH_BINARY)
            keep_mask = cv2.bitwise_not(mask)
            frame_gray = cv2.bitwise_and(frame_gray, keep_mask)


        # Compare the current frame against all frames in the window
        is_duplicate = False
        for existing_frame in comparison_window_hashes:
            score = step1_compare_frames(frame_gray, existing_frame, metric='mse', mask=None)
            if score < mse_threshold:
                is_duplicate = True
                break

        # If it's not a duplicate, keep it and add to the window
        if not is_duplicate:
            unique_filenames.append(filename)
            comparison_window_hashes.append(frame_gray)

        # Progress display
        progress = (idx + 1) / len(candidate_filenames)
        print(
            f"\rProcessing... [{int(progress*20)*'='}>{(19-int(progress*20))*' '}] "
            f"{int(progress*100)}% ({len(unique_filenames)} unique found)", end=""
        )

    print(f"\nStep 3 complete. Identified {len(unique_filenames)} unique slides from {len(candidate_filenames)} candidates.")

    # 5. Copy the final unique frames to the temp output folder
    for filename in unique_filenames:
        src_path = os.path.join(input_image_folder, filename)
        dst_path = os.path.join(temp_step3_candidates, filename)
        shutil.copy(src_path, dst_path)
    print(f"Temporary unique frames saved to: {temp_step3_candidates}")

    # 6. Copy final results to the main 'slides' directory, overwriting previous results
    print("Copying final unique slides to destination folder...")
    final_dest_folder = os.path.join(root_output_dir, "slides")
    copy_all_files(temp_step3_candidates, final_dest_folder)
    print(f"⏱️  step3_filter_with_sliding_window completed in {time.time() - start_time:.2f}s")




# --- Main Execution ---
if __name__ == "__main__":
    helper = Helper()
    total_start = time.time()

    # Configure file and directory paths
    video_path = "output/web_vpp/v8020/BIOL2301_lecture05_0609_8020_fps10.mp4"
    



    # Run the pipeline steps
    step1_basic_filter(video_path, processing_fps=1, enable_masking=True, speaker_mask_window_size=13, mse_threshold=50, manual_slide_bbox=None, skip_on_existing=True)
    
    # # Optional: If you want to find and hack a bounding box mask for slide area
    # step1_candidates_folder = os.path.join(video_path, "temp", "temp_step1_candidates")
    # slide_boundary = calculate_median_slide_bounding_box(step1_candidates_folder, num_images_to_process=200)


    step2_transition_filter(video_path, uniqueness_threshold=50, manual_slide_bbox=None, skip_on_existing=True)
    
    step3_filter_with_sliding_window(video_path, window_size=20, mse_threshold=50, skip_on_existing=True)
 

    print(f"\n✅ All steps completed in {time.time() - total_start:.2f} seconds.")




    # Other filtering to consider:
    # # Very short time duration, e.g., less than 2 seconds
    # # OCR text checking or transcript presence for those frames
    # # Intermediate ***blurred frames***, compare with edges and OCR text detection
    # # Hidden lines after explanation
