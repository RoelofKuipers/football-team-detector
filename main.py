import argparse
from pathlib import Path
from src.video_processor import VideoProcessor
from src.player_tracker import PlayerTracker

# Default paths
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"  # Default YOLO weights if no custom model found
CUSTOM_MODEL_PATH = "checkpoints/yolo_football.pt"  # Path to custom weights
OUTPUT_DIR = Path("output")  # Fixed output directory
INPUT_DIR = Path("data")  # Fixed input directory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Soccer Player Detection and Team Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="Path to input video file"
    )
    return parser.parse_args()


def get_model_path():
    """Check for custom model weights, otherwise use default YOLO weights"""
    custom_weights = Path(CUSTOM_MODEL_PATH)
    if custom_weights.exists():
        print(f"Using custom model weights: {custom_weights}")
        return str(custom_weights)
    else:
        raise ValueError(
            f"Custom weights not found at {custom_weights}, please check the path"
        )


def main():
    args = parse_args()
    # Setup paths
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Create directory structure
    OUTPUT_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)

    frames_dir = INPUT_DIR / "input_frames"  # Input frames go here
    output_frames_dir = OUTPUT_DIR / "frames"  # Annotated frames go here

    frames_dir.mkdir(exist_ok=True)
    output_frames_dir.mkdir(exist_ok=True)

    # Get appropriate model weights
    model_path = get_model_path()

    # Initialize components
    video_processor = VideoProcessor(
        video_path=input_path,
        frames_dir=frames_dir,
        output_frames_dir=output_frames_dir,  # Add this parameter
    )

    player_tracker = PlayerTracker(
        model_path=model_path,
        class_names=["Player", "Main Referee", "Side Referee", "GoalKeeper"],
    )

    # Process video
    print(f"Processing video: {input_path}")
    print(f"Saving output to: {OUTPUT_DIR}")

    # Extract frames if needed
    video_processor.extract_frames()

    # Run player tracking and team detection
    results = player_tracker.process_video(video_processor)

    # Save results and create output video
    video_processor.save_results(results)
    video_processor.save_video(
        frames_pattern="frame_*.jpg", output_name="match_processed.mp4"
    )

    print(f"\nProcessing complete! Output saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
