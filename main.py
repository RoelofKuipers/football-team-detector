import argparse
from pathlib import Path
from src.video_processor import VideoProcessor
from src.player_tracker import PlayerTracker

# Default paths
CUSTOM_MODEL_PATH = "checkpoints/yolo_football.pt"  # Path to custom weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Soccer Player Detection and Team Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Path to output directory"
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

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    frames_dir = output_dir / "input_frames"
    output_frames_dir = output_dir / "output_frames"

    frames_dir.mkdir(exist_ok=True)
    output_frames_dir.mkdir(exist_ok=True)

    model_path = get_model_path()

    video_processor = VideoProcessor(
        video_path=input_path,
        frames_dir=frames_dir,
        output_frames_dir=output_frames_dir,
    )

    player_tracker = PlayerTracker(
        model_path=model_path,
        class_names=["Player", "Main Referee", "Side Referee", "GoalKeeper"],
    )

    print(f"Processing video: {input_path}")
    print(f"Saving output to: {output_dir}")

    video_processor.extract_frames()
    results = player_tracker.process_video(video_processor)

    video_processor.save_results(results)
    video_processor.save_video(
        frames_pattern="frame_*.jpg", output_name="match_processed.mp4"
    )

    print(f"\nProcessing complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
