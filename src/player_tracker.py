from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.kit_classifier import KitClassifier
from src.yolo_model import YoloModel
from src.logger import setup_logger

logger = setup_logger(__name__)


class PlayerTracker:
    """
    Tracks and classifies players in football/soccer video footage.

    This class handles player detection, team classification based on kit colors,
    and visualization of results. It uses YOLOv8 for object detection and custom
    color-based classification for team assignment.

    Attributes:
        yolo_model (YoloModel): Model for player detection
        kit_classifier (KitClassifier): Classifier for team kit colors
        class_names (List[str]): List of class names to detect
        frame_detections (Dict): Stores detection results per frame
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        video_path: Optional[Union[str, Path]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize PlayerTracker with detection model and parameters.

        Args:
            model_path: Path to YOLO model weights
            video_path: Path to input video file (optional)
            class_names: List of class names to detect (e.g. ["Player", "Referee"])

        Raises:
            ValueError: If model_path is None or invalid
            ValueError: If class_names is not a list of strings
        """
        if not model_path:
            logger.error("model_path cannot be None")
            raise ValueError("model_path cannot be None")
        if class_names and not all(isinstance(name, str) for name in class_names):
            logger.error("class_names must be a list of strings")
            raise ValueError("class_names must be a list of strings")

        logger.info(f"Initializing PlayerTracker with model {model_path}")
        self.yolo_model = YoloModel(model_path, class_names)
        self.kit_classifier = KitClassifier()
        self.class_names = class_names
        self.frame_detections: Dict = {}

    def process_video(self, video_processor) -> Dict:
        """
        Process video to detect and classify players frame by frame.

        Args:
            video_processor: VideoProcessor instance for handling video I/O

        Returns:
            Dict containing frame-by-frame results with team counts and positions

        Raises:
            ValueError: If video_processor is None
            ValueError: If grass color extraction fails
            RuntimeError: If frame processing fails
        """
        if video_processor is None:
            logger.error("video_processor cannot be None")
            raise ValueError("video_processor cannot be None")

        logger.info("Starting video processing")
        logger.info("Detecting objects per frame index")

        # Stage 1: Extract grass color from first frame
        logger.info("Stage 1: Extracting grass color from first frame")
        frames = video_processor.get_frames_list()
        if not frames:
            logger.error("No frames found in video")
            raise ValueError("No frames found in video")

        logger.debug(f"Processing first frame: {frames[0]}")
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            logger.error("Could not read first frame")
            raise ValueError("Could not read first frame")

        self.kit_classifier.extract_grass_color(first_frame)

        # Stage 2: Process one frame at a time
        logger.info(
            "Stage 2: Processing frames: predicting football players, extracting kit colors"
        )
        all_kit_colors = []
        all_player_boxes = []
        self.frame_detections = {}

        for frame_idx, frame in video_processor.iter_frames():
            if frame is None:
                logger.error(f"Could not read frame {frame_idx}")
                raise RuntimeError(f"Could not read frame {frame_idx}")

            # Process current frame
            objects = self.yolo_model.detect(frame)
            if objects is None:
                logger.error(f"Detection failed for frame {frame_idx}")
                raise RuntimeError(f"Detection failed for frame {frame_idx}")

            # Extract bounding boxes and player images
            player_imgs, player_boxes = self._get_players_boxes(objects)
            if not player_imgs or not player_boxes:
                continue

            # Extract kit colors in player images
            kit_colors = self.kit_classifier._get_kits_colors(player_imgs)
            if not kit_colors:
                continue

            self.frame_detections[frame_idx] = {
                "yolo_result": objects,
                "player_boxes": player_boxes,
                "kit_colors": kit_colors,
            }

            all_kit_colors.extend(kit_colors)
            all_player_boxes.extend(player_boxes)

        if not all_kit_colors:
            logger.error("No kit colors found in video")
            raise ValueError("No kit colors found in video")

        # Stage 3: Train Team Classifier
        logger.info("Stage 3: Training team classifier...")
        self.kit_classifier.train_classifier(all_kit_colors)
        all_team_labels = self.kit_classifier.classify_teams(all_kit_colors)
        self.kit_classifier.determine_left_team(all_player_boxes, all_team_labels)

        # Stage 4: Process Each Frame
        logger.info("Stage 4: Generating final output...")
        results = {}
        current_box_idx = 0

        for frame_idx, data in self.frame_detections.items():
            num_players = len(data["player_boxes"])
            frame_team_labels = all_team_labels[
                current_box_idx : current_box_idx + num_players
            ]

            # Generate visualization and counts
            annotated_frame, team_left_count, team_right_count = self._draw_team_boxes(
                data["yolo_result"],
                data["player_boxes"],
                frame_team_labels,
            )

            results[frame_idx] = {
                "team_counts": {"left": team_left_count, "right": team_right_count}
            }

            video_processor.save_frame(annotated_frame, f"frame_{frame_idx:04d}.jpg")

            current_box_idx += num_players

        if not results:
            logger.error("No results generated")
            raise ValueError("No results generated")

        logger.info("Video processing completed successfully")
        return results

    def _get_players_boxes(self, result) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract player bounding boxes and cropped images from YOLO detection results.

        Args:
            result: YOLO detection result object

        Returns:
            Tuple containing:
                - List of cropped player images as numpy arrays
                - List of bounding box coordinates

        Raises:
            ValueError: If result is None
            ValueError: If player class name not found in class_names
            ValueError: If no players detected in frame
        """
        if result is None:
            logger.error("Detection result cannot be None")
            raise ValueError("Detection result cannot be None")
        if not self.class_names:
            logger.error("class_names not initialized")
            raise ValueError("class_names not initialized")

        players_imgs = []
        players_boxes = []

        player_label_name = "Player"
        if player_label_name not in self.class_names:
            logger.error(
                f"Player label {player_label_name} not found in class names, got {self.class_names}"
            )
            raise ValueError(
                f"Player label {player_label_name} not found in class names, got {self.class_names}"
            )

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            if x1 >= x2 or y1 >= y2:
                continue
            player_img = result.orig_img[y1:y2, x1:x2]
            if player_img.size == 0:
                continue
            players_imgs.append(player_img)
            players_boxes.append(box)

        if not players_imgs:
            return [], []

        return players_imgs, players_boxes

    def _draw_team_boxes(
        self,
        yolo_result,
        player_boxes: List[np.ndarray],
        team_labels: List[int],
    ) -> Tuple[np.ndarray, int, int]:
        """
        Draw bounding boxes and labels on frame for visualization.

        Args:
            yolo_result: YOLO detection result object
            player_boxes: List of player bounding boxes
            team_labels: List of team labels for each player

        Returns:
            Tuple containing:
                - Annotated frame as numpy array
                - Count of left team players
                - Count of right team players

        Raises:
            ValueError: If yolo_result is None
            ValueError: If player_boxes is empty
            ValueError: If team_labels is empty
            ValueError: If lengths don't match
        """
        if yolo_result is None:
            raise ValueError("yolo_result cannot be None")
        if len(player_boxes) != len(team_labels):
            raise ValueError("player_boxes and team_labels must have same length")

        team_colors = {
            0: (0, 0, 255),  # Red for left team
            1: (255, 0, 0),  # Blue for right team
            2: (0, 255, 0),  # Green for referees
            10: (255, 0, 255),  # Purple for others
        }

        annotated_frame = yolo_result.orig_img.copy()

        # Dictionary to store kit colors per player ID
        player_team_labels = defaultdict(list)

        # Counter for players per team
        team_counts = defaultdict(int)

        # Draw boxes and text efficiently
        for box_id, box in enumerate(yolo_result.boxes):
            track_id = None
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id.numpy()[0])

            # get the team label for this box
            team = team_labels[box_id]

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, player_boxes[box_id].xyxy[0].numpy())

            # get the label name for this box
            label = int(box.cls.numpy()[0])
            label_name = yolo_result.names.get(label)
            if label_name is None:
                continue  # Skip if class name not found

            if label_name == "Player":
                if track_id is not None:
                    player_team_labels[track_id].append(team)
                    team = np.argmax(np.bincount(player_team_labels[track_id]))

                if team == self.kit_classifier.left_team_label:
                    team_nr = 0
                elif team == 1 - self.kit_classifier.left_team_label:
                    team_nr = 1
                else:
                    team_nr = 2

                team_counts[team_nr] += 1

            elif label_name == "GoalKeeper":
                if x1 < 0.5 * annotated_frame.shape[1]:
                    team_nr = 0
                else:
                    team_nr = 1
                team_counts[team_nr] += 1
            elif label_name in ["Main Referee", "Side Referee"]:
                team_nr = 2
                team_counts[team_nr] += 1
            else:
                team_nr = 10
                team_counts[team_nr] += 1

            color = team_colors[team_nr]

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)

            # Draw text with track ID if available
            if team_nr == 0:
                label = "Team Left"
            elif team_nr == 1:
                label = "Team Right"
            elif team_nr == 2:
                label = "Referee"
            else:
                label = label_name
            if track_id is not None:
                label += f" - player {track_id}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        # Draw team counts with background
        y_offset = 80
        x_offset = 80
        padding = 20
        font_scale = 1.2

        for team_nr in sorted(team_counts.keys()):
            if team_nr == self.kit_classifier.left_team_label:
                count_text = f"Team Left: {team_counts[team_nr]} players"
            elif team_nr == 1 - self.kit_classifier.left_team_label:
                count_text = f"Team Right: {team_counts[team_nr]} players"
            else:
                continue

            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )

            # Draw background rectangle
            cv2.rectangle(
                annotated_frame,
                (x_offset - padding, y_offset - text_height - padding),
                (x_offset + text_width + padding, y_offset + padding),
                (200, 200, 200),  # Light grey color
                -1,
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                count_text,
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                team_colors[team_nr],
                2,
            )
            y_offset += text_height + padding * 2

        return annotated_frame, team_counts[0], team_counts[1]
