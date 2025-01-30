from pathlib import Path
import cv2
from tqdm import tqdm
import json
from typing import Union, Dict, Any, List, Iterator, Tuple

from src.player_tracker import PlayerTracker
import shutil

from src.logger import setup_logger

logger = setup_logger(__name__)


class VideoProcessor:
    def __init__(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> None:
        """Initialize VideoProcessor with paths for video processing.

        Args:
            video_path: Path to input video file
            output_dir: Directory to store output files

        Raises:
            ValueError: If video_path does not exist or is not a file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists() or not self.video_path.is_file():
            logger.error(f"Video path {video_path} does not exist or is not a file")
            raise ValueError(f"Video path {video_path} does not exist or is not a file")

        self.output_dir = Path(output_dir)
        self.frames_dir = output_dir / "input_frames"
        self.output_frames_dir = output_dir / "output_frames"

        self.frames_dir.mkdir(exist_ok=True)
        self.output_frames_dir.mkdir(exist_ok=True)

        self.frame_count = 0

        self.frame_count = 0
        self.fps: int = 0
        self.width: int = 0
        self.height: int = 0
        self._get_video_info()
        logger.info(f"Initialized VideoProcessor for {video_path}")

    def _get_video_info(self) -> None:
        """Get video metadata.

        Raises:
            RuntimeError: If video file cannot be opened
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file {self.video_path}")
            raise RuntimeError(f"Could not open video file {self.video_path}")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        logger.info(f"Video info: {self.width}x{self.height} @ {self.fps}fps")

    def extract_frames(self) -> None:
        """Extract frames from video.

        Raises:
            RuntimeError: If frames cannot be extracted
        """
        if (
            not self.frames_dir.exists()
            or len(list(self.frames_dir.glob("*.jpg"))) == 0
        ):
            logger.info("Starting frame extraction")
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file {self.video_path}")
                raise RuntimeError(f"Could not open video file {self.video_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = self.frames_dir / f"frame_{self.frame_count:04d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    logger.error(f"Could not write frame to {frame_path}")
                    raise RuntimeError(f"Could not write frame to {frame_path}")
                self.frame_count += 1
            cap.release()
            logger.info(f"Extracted {self.frame_count} frames")

    def get_frames_list(self) -> List[Path]:
        """Get sorted list of frame paths.

        Returns:
            List of paths to frame files

        Raises:
            RuntimeError: If no frames are found
        """
        frames = list(self.frames_dir.glob("*.jpg"))
        if not frames:
            logger.error(f"No frames found in {self.frames_dir}")
            raise RuntimeError(f"No frames found in {self.frames_dir}")
        logger.debug(f"Found {len(frames)} frames")
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
                logger.error("max_frames must be a positive integer")
                raise ValueError("max_frames must be a positive integer")
            frames = frames[:max_frames]
            logger.info(f"Processing {max_frames} frames")

        for frame_path in tqdm(frames, desc="Processing frames"):
            frame_idx = int(frame_path.stem.split("_")[1])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.error(f"Could not read frame {frame_path}")
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
            logger.error("Frame cannot be None")
            raise ValueError("Frame cannot be None")

        self.output_frames_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_frames_dir / filename
        if not cv2.imwrite(str(output_path), frame):
            logger.error(f"Could not save frame to {output_path}")
            raise RuntimeError(f"Could not save frame to {output_path}")
        logger.debug(f"Saved frame to {output_path}")
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
            logger.error(f"No processed frames found in {self.output_frames_dir}")
            raise ValueError(f"No processed frames found in {self.output_frames_dir}")

        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            logger.error(f"Could not read first frame {frame_files[0]}")
            raise RuntimeError(f"Could not read first frame {frame_files[0]}")

        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        if not out.isOpened():
            logger.error(f"Could not create video writer for {output_path}")
            raise RuntimeError(f"Could not create video writer for {output_path}")

        logger.info("Creating output video...")
        try:
            for frame_path in tqdm(frame_files):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.error(f"Could not read frame {frame_path}")
                    raise RuntimeError(f"Could not read frame {frame_path}")
                out.write(frame)
        finally:
            out.release()

        logger.info(f"Video saved to {output_path}")
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
            logger.error("Results dictionary cannot be empty")
            raise ValueError("Results dictionary cannot be empty")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_name

        try:
            with open(output_path, "w") as f:
                json.dump(results, f)
        except Exception as e:
            logger.error(f"Could not save results to {output_path}: {str(e)}")
            raise RuntimeError(f"Could not save results to {output_path}: {str(e)}")

        logger.info(f"Results saved to {output_path}")
        return output_path


# For now keep it as a standalone function rather than adding it to the VideoProcessor class because:
# - It orchestrates multiple components (VideoProcessor and PlayerTracker)
# - It follows the Single Responsibility Principle - VideoProcessor should focus on video I/O operations
# - It's more flexible for testing and reuse
# - It acts more as a workflow coordinator than a core video processing functionality
def process_football_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_path: Union[str, Path],
    cleanup_frames: bool = True,
) -> Tuple[Dict, Path]:
    """Process a football video and return results and output path.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        model_path: Path to YOLO model weights
        cleanup_frames: Whether to remove input frames after processing

    Returns:
        Tuple containing:
            - Dictionary of processing results
            - Path to output video file
    """
    logger.info(f"Starting video processing for {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    video_processor = VideoProcessor(
        video_path=video_path,
        output_dir=output_dir,
    )

    player_tracker = PlayerTracker(
        model_path=model_path,
        class_names=["Player", "Main Referee", "Side Referee", "GoalKeeper"],
    )

    video_processor.extract_frames()
    results = player_tracker.process_video(
        video_processor=video_processor, cleanup_frames=cleanup_frames
    )

    results_path = video_processor.save_results(results)
    output_video_path = video_processor.save_video(
        frames_pattern="frame_*.jpg", output_name="match_processed.mp4"
    )

    logger.info("Video processing completed successfully")
    return results, output_video_path
