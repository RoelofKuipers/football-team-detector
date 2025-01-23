from pathlib import Path
import cv2
from tqdm import tqdm
import json
from typing import Union, Dict, Any, List, Iterator, Tuple


class VideoProcessor:
    def __init__(
        self,
        video_path: Union[str, Path],
        frames_dir: Union[str, Path],
        output_frames_dir: Union[str, Path],
    ) -> None:
        """Initialize VideoProcessor with paths for video processing.

        Args:
            video_path: Path to input video file
            frames_dir: Directory to store extracted frames
            output_frames_dir: Directory to store processed frames

        Raises:
            ValueError: If video_path does not exist or is not a file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists() or not self.video_path.is_file():
            raise ValueError(f"Video path {video_path} does not exist or is not a file")

        self.frames_dir = Path(frames_dir)  # for input frames
        self.output_frames_dir = Path(output_frames_dir)  # for annotated frames
        self.output_dir = Path("output")  # base output directory

        self.frame_count = 0
        self.fps: int = 0
        self.width: int = 0
        self.height: int = 0
        self._get_video_info()

    def _get_video_info(self) -> None:
        """Get video metadata.

        Raises:
            RuntimeError: If video file cannot be opened
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file {self.video_path}")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def extract_frames(self) -> None:
        """Extract frames from video.

        Raises:
            RuntimeError: If frames cannot be extracted
        """
        if (
            not self.frames_dir.exists()
            or len(list(self.frames_dir.glob("*.jpg"))) == 0
        ):
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file {self.video_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = self.frames_dir / f"frame_{self.frame_count:04d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write frame to {frame_path}")
                self.frame_count += 1
            cap.release()

    def get_frames_list(self) -> List[Path]:
        """Get sorted list of frame paths.

        Returns:
            List of paths to frame files

        Raises:
            RuntimeError: If no frames are found
        """
        frames = list(self.frames_dir.glob("*.jpg"))
        if not frames:
            raise RuntimeError(f"No frames found in {self.frames_dir}")
        return sorted(frames, key=lambda x: int(x.stem.split("_")[1]))

    def iter_frames(self, max_frames: int = None) -> Iterator[Tuple[int, Any]]:
        """Iterate over frames, returning (index, frame) pairs.

        Args:
            max_frames: Maximum number of frames to process

        Yields:
            Tuple of (frame_index, frame_data)

        Raises:
            RuntimeError: If frames cannot be read
        """
        frames = self.get_frames_list()
        if max_frames is not None:
            if not isinstance(max_frames, int) or max_frames <= 0:
                raise ValueError("max_frames must be a positive integer")
            frames = frames[:max_frames]

        for frame_path in tqdm(frames, desc="Processing frames"):
            frame_idx = int(frame_path.stem.split("_")[1])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame {frame_path}")
            yield frame_idx, frame

    def save_frame(self, frame: Any, filename: str) -> Path:
        """Save a processed frame.

        Args:
            frame: Frame data to save
            filename: Output filename

        Returns:
            Path to saved frame

        Raises:
            RuntimeError: If frame cannot be saved
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        self.output_frames_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_frames_dir / filename
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not save frame to {output_path}")
        return output_path

    def save_video(self, frames_pattern: str, output_name: str = "output.mp4") -> Path:
        """Create video from processed frames.

        Args:
            frames_pattern: Pattern to match frame files
            output_name: Name of output video file

        Returns:
            Path to output video file

        Raises:
            ValueError: If no processed frames are found
            RuntimeError: If video cannot be created
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_name

        frame_files = sorted(self.output_frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            raise ValueError(f"No processed frames found in {self.output_frames_dir}")

        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise RuntimeError(f"Could not read first frame {frame_files[0]}")

        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Could not create video writer for {output_path}")

        print("Creating output video...")
        try:
            for frame_path in tqdm(frame_files):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Could not read frame {frame_path}")
                out.write(frame)
        finally:
            out.release()

        print(f"Video saved to {output_path}")
        return output_path

    def save_results(
        self, results: Dict[str, Any], output_name: str = "teams_per_frame.json"
    ) -> Path:
        """Save detection results to JSON.

        Args:
            results: Results dictionary to save
            output_name: Name of output JSON file

        Returns:
            Path to output JSON file

        Raises:
            ValueError: If results is empty
            RuntimeError: If results cannot be saved
        """
        if not results:
            raise ValueError("Results dictionary cannot be empty")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_name

        try:
            with open(output_path, "w") as f:
                json.dump(results, f)
        except Exception as e:
            raise RuntimeError(f"Could not save results to {output_path}: {str(e)}")

        print(f"Results saved to {output_path}")
        return output_path
