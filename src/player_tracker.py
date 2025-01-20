from collections import defaultdict
import numpy as np
import cv2
from sklearn.cluster import KMeans
import tqdm
from pathlib import Path
from tqdm import tqdm
from src.kit_classifier import KitClassifier
from src.yolo_model import YoloModel


class PlayerTracker:
    def __init__(
        self,
        model_path=None,
        video_path=None,
        class_names=None,
    ):
        """
        Initialize TeamDetector with model and parameters

        Args:
            model_path: Path to YOLO model (if None, loads default)
            n_teams: Number of teams to detect (default 2)
        """
        self.yolo_model = YoloModel(model_path, class_names)
        self.kit_classifier = KitClassifier()
        self.class_names = class_names

    def process_video(self, video_processor):
        print("Detecting objects per frame index")

        # Stage 1: Extract grass color from first frame
        print("Stage 1: Extracting grass color from first frame")
        frames = video_processor.get_frames_list()
        print(frames[0])
        first_frame = cv2.imread(str(frames[0]))  # Read first frame directly
        self.kit_classifier.extract_grass_color(first_frame)

        # Step 2: Process one frame at a time
        all_kit_colors = []
        all_player_boxes = []
        self.frame_detections = {}
        for frame_idx, frame in video_processor.iter_frames():
            # Process current frame
            objects = self.yolo_model.detect(frame)

            # Extract bounding boxes and player images
            player_imgs, player_boxes = self._get_players_boxes(objects)
            # Extract kit colors in player images
            kit_colors = self.kit_classifier._get_kits_colors(player_imgs)

            self.frame_detections[frame_idx] = {
                "yolo_result": objects,
                "player_boxes": player_boxes,
                "kit_colors": kit_colors,
            }

            all_kit_colors.extend(kit_colors)
            all_player_boxes.extend(player_boxes)

        # Stage 3: Train Team Classifier
        print("Stage 3: Training team classifier...")
        self.kit_classifier.train_classifier(all_kit_colors)
        all_team_labels = self.kit_classifier.classify_teams(all_kit_colors)
        self.kit_classifier.determine_left_team(all_player_boxes, all_team_labels)

        # Stage 4: Process Each Frame
        print("Stage 4: Generating final output...")
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

        return results

    def _get_players_boxes(self, result):
        """Get players boxes from YOLO result"""
        # this can only run after
        players_imgs = []
        players_boxes = []

        player_label_name = "Player"
        if player_label_name not in self.class_names:
            raise ValueError(
                f"Player label {player_label_name} not found in class names, got {self.class_names}"
            )
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            player_img = result.orig_img[y1:y2, x1:x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
        players_imgs = players_imgs
        players_boxes = players_boxes
        return (
            players_imgs,
            players_boxes,
        )

    def _draw_team_boxes(
        self,
        yolo_result,
        player_boxes,
        team_labels,
    ):

        team_colors = {
            0: (0, 0, 255),
            1: (255, 0, 0),
            2: (0, 255, 0),
            10: (255, 0, 255),
        }  # 10 sohuld be purple

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
            label_name = yolo_result.names.get(
                label
            )  # Use .get() for safe dictionary access
            if label_name is None:
                continue  # Skip if class name not found

            if label_name == "Player":
                # # Store kit color for this player ID
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
            elif label_name == "Main Referee":
                team_nr = 2
                team_counts[team_nr] += 1
            elif label_name == "Side Referee":
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
                label = f"Team Left"
            elif team_nr == 1:
                label = f"Team Right"
            elif team_nr == 2:
                label = f"Referee"
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
