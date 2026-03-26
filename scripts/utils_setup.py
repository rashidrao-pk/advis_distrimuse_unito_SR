import os, json, math
import cv2

#-------------------------------------------------------------------------------------------------
#    extract_crop_resize_save
#-------------------------------------------------------------------------------------------------
# ------------------------- main routine -------------------------
def ping_setup():
    print('-'*100)
    print('SETUP DONE')
    print('-'*100)


def extract_crop_resize_save(
    video_path: str,
    output_dir: str,
    *,
    static_mask_path: str = None,
    mask_template: str = None,         # e.g., r"D:\masks\mask_{i:05d}.png"
    start_frame: int = 0,
    end_frame: int = None,
    step: int = 1,
    keep_aspect: bool = True,
    save_masked_full: bool = False,    # also save full-size masked frame (optional)
    output_ext: str = "png"
):
    """
    For each frame:
      - load corresponding mask (static or per-frame)
      - apply mask, crop to bbox
      - resize to 128x128
      - save as image
    """
    if not os.path.exists(video_path) or not os.path.exists(mask_template):
        assert "ERROR"

    print(os.path.exists(mask_template), mask_template)
    os.makedirs(output_dir, exist_ok=True)
    print('Output dir:', output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if end_frame is None or end_frame > total > 0:
        end_frame = total

    # Load static mask once (if provided)
    static_mask = None
    if static_mask_path:
        static_mask = _ensure_gray(cv2.imread(static_mask_path, cv2.IMREAD_GRAYSCALE))
        if static_mask is None:
            raise RuntimeError(f"Failed to read static mask: {static_mask_path}")

    # If we know the resolution, pre-resize static mask to match
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('='*120)
    # print('vid_w: ', vid_w)
    # print('vid_h: ', vid_h)

    if static_mask is not None and vid_w and vid_h:
        if static_mask.shape[:2] != (vid_h, vid_w):
            static_mask = cv2.resize(static_mask, (vid_w, vid_h), interpolation=cv2.INTER_NEAREST)

    saved = 0
    frame_idx = 0

    # Position to start_frame (fast-forward)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    pbar_total = (end_frame - start_frame) // step if (end_frame and step > 0) else None
    with tqdm(total=pbar_total, desc="Cropping+Resizing frames", unit="frame") as pbar:
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % step == 0:
                # Decide mask for this frame
                mask_bin = static_mask
                if mask_bin is None:
                    if mask_template is None:
                        raise ValueError("Provide either static_mask_path OR mask_template.")
                    # Support {i} or {frame} variables in template
                    mask_path = mask_template.format(i=frame_idx, idx=frame_idx, frame=frame_idx)
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        print('m -',m.shape)
                        # If mask missing, skip this frame gracefully
                        pbar.update(1)
                        frame_idx += 1
                        continue
                    # Ensure same size as frame
                    if m.shape[:2] != frame.shape[:2]:
                        m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_bin = _ensure_gray(m)

                # Apply mask + crop
                cropped, masked_full = _crop_with_mask(frame, mask_bin)

                if cropped is not None:
                    # Resize to 128x128
                    resized = _resize_128(cropped, keep_aspect=keep_aspect, target=(128, 128))
                    out_name = os.path.join(output_dir, f"crop_{frame_idx:05d}.{output_ext}")
                    cv2.imwrite(out_name, resized)
                    saved += 1

                if save_masked_full:
                    mf_name = os.path.join(output_dir, f"masked_{frame_idx:05d}.{output_ext}")
                    cv2.imwrite(mf_name, masked_full)

                pbar.update(1)
            # break
            frame_idx += 1

    cap.release()
    print(f"✅ Saved {saved} cropped 128x128 frames to: {output_dir}")