import argparse
from pathlib import Path
from src.video_processor import process_football_video
import sys

# Default paths
CUSTOM_MODEL_PATH = "checkpoints/yolo_football.pt"  # Path to custom weights


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Soccer Player Detection and Team Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Path to output directory"
    )
    return parser.parse_args(args)


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


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parsed_args = parse_args(args)

    input_path = Path(parsed_args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = Path(parsed_args.output)
    model_path = get_model_path()

    print(f"Processing video: {input_path}")
    print(f"Saving output to: {output_dir}")

    results, output_video = process_football_video(
        video_path=input_path, output_dir=output_dir, model_path=model_path
    )

    print(f"\nProcessing complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
