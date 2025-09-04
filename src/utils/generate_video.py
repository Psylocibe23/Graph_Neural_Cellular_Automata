import cv2
import os
import glob
from natsort import natsorted


def make_video(growth_dir, attn_dir, out_path, fps=12):
    # Get sorted frame lists
    growth_frames = natsorted(glob.glob(os.path.join(growth_dir, 'frame_*.png')))
    attn_frames = natsorted(glob.glob(os.path.join(attn_dir, 'attn_*.png')))
    # Use the minimum number of frames
    num_frames = min(len(growth_frames), len(attn_frames))
    assert num_frames > 0, "No frames found!"

    # Read one to get size
    img_g = cv2.imread(growth_frames[0])
    img_a = cv2.imread(attn_frames[0])
    height = max(img_g.shape[0], img_a.shape[0])
    width = img_g.shape[1] + img_a.shape[1]

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for gf, af in zip(growth_frames, attn_frames):
        img_g = cv2.imread(gf)
        img_a = cv2.imread(af)
        # Resize to same height if needed
        if img_g.shape != img_a.shape:
            img_a = cv2.resize(img_a, (img_g.shape[1], img_g.shape[0]))
        # Concatenate side-by-side
        frame = cv2.hconcat([img_g, img_a])
        out.write(frame)

    out.release()
    print(f"Saved video to {out_path}")

def make_video_growth_only(growth_dir, out_path, fps=12):
    # Get sorted frame list
    growth_frames = natsorted(glob.glob(os.path.join(growth_dir, 'combo_*.png')))
    num_frames = len(growth_frames)
    assert num_frames > 0, "No frames found!"
    img_g = cv2.imread(growth_frames[0])
    height = img_g.shape[0]
    width = img_g.shape[1]

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for gf in growth_frames:
        img_g = cv2.imread(gf)
        out.write(img_g)
    out.release()
    print(f"Saved video to {out_path}")



if __name__ == "__main__":
    #base_dir = "outputs/graph_augmented/growth_video/gecko_graphaug"
    #growth_dir = base_dir
    #attn_dir = os.path.join(base_dir, "attention")
    #out_path = os.path.join(base_dir, "side_by_side.mp4")
    #make_video(growth_dir, attn_dir, out_path, fps=12)
    base_dir = "outputs/graphaug_nca/test_attention/gecko"
    out_path = os.path.join(base_dir, "growth_video.mp4")
    make_video_growth_only(base_dir, out_path, fps=12)
