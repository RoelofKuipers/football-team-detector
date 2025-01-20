from pathlib import Path
import cv2
from tqdm import tqdm
import json


class VideoProcessor:
    def __init__(self, video_path, frames_dir, output_frames_dir):
        self.video_path = Path(video_path)
        self.frames_dir = Path(frames_dir)  # for input frames
        self.output_frames_dir = Path(output_frames_dir)  # for annotated frames
        self.output_dir = Path("output")  # base output directory

        self.frame_count = 0
        self.fps = None
        self.width = None
        self.height = None
        self._get_video_info()

    def _get_video_info(self):
        """Get video metadata"""
        cap = cv2.VideoCapture(str(self.video_path))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def extract_frames(self):
        """Extract frames from video"""
        if (
            not self.frames_dir.exists()
            or len(list(self.frames_dir.glob("*.jpg"))) == 0
        ):
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(self.video_path))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    str(self.frames_dir / f"frame_{self.frame_count:04d}.jpg"), frame
                )
                self.frame_count += 1
            cap.release()

    def get_frames_list(self):
        """Get sorted list of frame paths"""
        frames = list(self.frames_dir.glob("*.jpg"))
        return sorted(frames, key=lambda x: int(x.stem.split("_")[1]))

    def iter_frames(self, max_frames=None):
        """Iterate over frames, returning (index, frame) pairs"""
        frames = self.get_frames_list()
        if max_frames:
            frames = frames[:max_frames]

        for frame_path in tqdm(frames, desc="Processing frames"):
            frame_idx = int(frame_path.stem.split("_")[1])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
            yield frame_idx, frame

    def save_frame(self, frame, filename):
        """Save a processed frame"""
        output_path = self.output_frames_dir / filename
        cv2.imwrite(str(output_path), frame)
        return output_path

    def save_video(self, frames_pattern, output_name="output.mp4"):
        """Create video from processed frames"""
        output_path = self.output_dir / output_name

        # Get frames from output_frames_dir
        frame_files = sorted(self.output_frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            raise ValueError("No processed frames found")

        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

        # Write frames to video
        print("Creating output video...")
        for frame_path in tqdm(frame_files):
            frame = cv2.imread(str(frame_path))
            out.write(frame)

        out.release()
        print(f"Video saved to {output_path}")
        return output_path

    def save_results(self, results, output_name="teams_per_frame.json"):
        """Save detection results to JSON"""
        output_path = self.output_dir / output_name
        with open(output_path, "w") as f:
            json.dump(results, f)
        print(f"Results saved to {output_path}")
        return output_path
