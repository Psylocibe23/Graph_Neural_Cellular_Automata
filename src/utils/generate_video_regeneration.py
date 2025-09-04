import os
import re
import sys
import glob
try:
    import cv2
except Exception as e:
    raise SystemExit(
        "OpenCV (cv2) is required. Try: pip install opencv-python\n"
        f"Import error: {e}"
    )

def _nat_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _default_root():
    candidates = [
        r"C:\Users\sprea\Desktop\pythonProject\GNN_NCA\outputs\graphaug_nca\test_regrowth\gecko",
        "/mnt/c/Users/sprea/Desktop/pythonProject/GNN_NCA/outputs/graphaug_nca/test_regrowth/gecko",
        os.path.abspath(os.path.join(os.path.dirname(__file__),
            "..", "..", "outputs", "graphaug_nca", "test_regrowth", "gecko")),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[-1]

def collect_damage_dirs(root: str):
    """All subfolders that contain at least one combo_*.png."""
    dmg_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(fn.startswith("combo_") and fn.endswith(".png") for fn in filenames):
            dmg_dirs.append(dirpath)
    dmg_dirs.sort(key=_nat_key)
    return dmg_dirs

def make_video_from_combo(folder: str, fps: int = 20,
                          pattern: str = "combo_*.png",
                          outname: str = "combo.mp4") -> bool:
    frames = sorted(glob.glob(os.path.join(folder, pattern)), key=_nat_key)
    if not frames:
        return False

    first = cv2.imread(frames[0])
    if first is None:
        print(f"[warn] unreadable first frame in {folder}")
        return False

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(folder, outname)
    os.makedirs(folder, exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[error] could not open VideoWriter for {out_path}")
        return False

    written = 0
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            print(f"[warn] skipped unreadable: {fp}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        written += 1

    writer.release()
    print(f"[video] {out_path}  ({written} frames @ {fps} fps)")
    return written > 0

def main():
    root = os.environ.get("GNN_NCA_REGROWTH_ROOT") or _default_root()
    fps = int(os.environ.get("GNN_NCA_VIDEO_FPS", "20"))

    if not os.path.isdir(root):
        print(f"[error] root not found: {root}")
        sys.exit(1)

    print(f"[info] scanning damage folders under:\n  {root}")
    dmg_dirs = collect_damage_dirs(root)
    if not dmg_dirs:
        print("[info] no folders with combo_*.png were found.")
        sys.exit(0)

    made = 0
    for d in dmg_dirs:
        if make_video_from_combo(d, fps=fps):
            made += 1

    print(f"[done] built videos in {made} folder(s).")

if __name__ == "__main__":
    main()
