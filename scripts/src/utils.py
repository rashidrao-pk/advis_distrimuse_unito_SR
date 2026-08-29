import cv2
import numpy as np
from pathlib import Path


def motion_score(prev_frame,
                 curr_frame,
                 mask,
                 pixel_threshold=20):

    # grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # small blur to suppress camera/image noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # absolute frame difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # threshold meaningful pixel changes
    motion = (diff > pixel_threshold).astype(np.uint8)

    # restrict to safety ROI
    roi_motion = motion * (mask > 0)

    roi_pixels = np.count_nonzero(mask)

    if roi_pixels == 0:
        return 0.0

    changed_pixels = np.count_nonzero(roi_motion)

    return changed_pixels / roi_pixels



def analyze_video(video_path,
                  safety_masks,
                  motion_threshold=0.01):

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()

    if not ret:
        return None

    results = {
        name: []
        for name in safety_masks
    }

    frame_id = 1

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        for area_name, mask in safety_masks.items():

            score = motion_score(
                prev_frame,
                frame,
                mask
            )

            active = score > motion_threshold

            results[area_name].append({
                "frame": frame_id,
                "motion_score": score,
                "active": active
            })

        prev_frame = frame.copy()

        frame_id += 1

    cap.release()

    return results