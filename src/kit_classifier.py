from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


class KitClassifier:
    """Classifier for detecting and categorizing sports team kits/uniforms in images."""

    def __init__(self) -> None:
        """Initialize the KitClassifier.

        The classifier needs to be trained on kit colors and have grass color extracted
        before it can classify teams.
        """
        self.grass_color: Optional[np.ndarray] = None
        self.kits_classifier: Optional[KMeans] = None
        self.left_team_label: Optional[int] = None
        self.teams: Optional[np.ndarray] = None
        self.kits_colors: Optional[List[np.ndarray]] = None

    def extract_grass_color(self, frame: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Extract the dominant grass color from a frame.

        Args:
            frame: Input frame as a file path or numpy array

        Returns:
            np.ndarray: BGR color values of detected grass

        Raises:
            ValueError: If no grass is found in the image or frame is invalid
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        self.grass_color = self._get_grass_color(frame)
        return self.grass_color

    def get_kit_colors(
        self, player_imgs: List[np.ndarray], resize_percent: int = 50
    ) -> List[np.ndarray]:
        """Extract kit colors from cropped player images.

        Args:
            player_imgs: List of cropped player image arrays

        Returns:
            List[np.ndarray]: BGR color values for each player's kit

        Raises:
            ValueError: If grass color has not been extracted first or player_imgs is invalid
        """
        if self.grass_color is None:
            raise ValueError("Must extract grass color first")
        if not player_imgs:
            raise ValueError("Player images list cannot be empty")
        if not all(isinstance(img, np.ndarray) for img in player_imgs):
            raise ValueError("All player images must be numpy arrays")

        return self._get_kits_colors(player_imgs, resize_percent=resize_percent)

    def train_classifier(self, all_kit_colors: List[np.ndarray]) -> KMeans:
        """Train the classifier on extracted kit colors.

        Args:
            all_kit_colors: List of BGR kit colors to train on

        Returns:
            KMeans: Trained classifier model

        Raises:
            ValueError: If kit colors are invalid
        """
        if not all_kit_colors:
            raise ValueError("Kit colors list cannot be empty")
        if not all(isinstance(color, np.ndarray) for color in all_kit_colors):
            raise ValueError("All kit colors must be numpy arrays")

        self.kits_classifier = self._get_kits_classifier(all_kit_colors)
        return self.kits_classifier

    def classify_teams(self, kit_colors: List[np.ndarray]) -> np.ndarray:
        """Classify kit colors into teams.

        Args:
            kit_colors: List of BGR kit colors to classify

        Returns:
            np.ndarray: Team labels for each kit color

        Raises:
            ValueError: If classifier has not been trained or kit colors are invalid
        """
        if self.kits_classifier is None:
            raise ValueError("Must train classifier first")
        if not kit_colors:
            raise ValueError("Kit colors list cannot be empty")
        if not all(isinstance(color, np.ndarray) for color in kit_colors):
            raise ValueError("All kit colors must be numpy arrays")

        return self._classify_kits(kit_colors)

    def determine_left_team(
        self, players_boxes: List[Any], team_labels: np.ndarray
    ) -> int:
        """Determine which team is on the left side of the frame.

        Args:
            players_boxes: Bounding boxes for detected players
            team_labels: Team classification labels

        Returns:
            int: Label of the team on the left (0 or 1)

        Raises:
            ValueError: If player boxes or team labels are empty/invalid
        """
        if not players_boxes:
            raise ValueError("Players boxes list cannot be empty")
        if not isinstance(team_labels, np.ndarray):
            raise ValueError("Team labels must be a numpy array")
        if len(team_labels) == 0:
            raise ValueError("Team labels array cannot be empty")

        self.left_team_label = self._get_left_team_label(players_boxes, team_labels)
        return self.left_team_label

    def _get_grass_color(self, frame: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Extract grass color from frame using HSV thresholding.

        Args:
            frame: Input frame as file path or numpy array

        Returns:
            np.ndarray: BGR color values of detected grass

        Raises:
            ValueError: If no grass is found or image cannot be loaded
        """
        if isinstance(frame, (str, Path)):
            img = cv2.imread(str(frame))
            if img is None:
                raise ValueError(f"Failed to load image from {frame}")
        else:
            if not isinstance(frame, np.ndarray):
                raise ValueError("Frame must be a file path or numpy array")
            img = frame

        # Convert and threshold in one step
        mask = cv2.inRange(
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
            np.array([35, 50, 50]),
            np.array([75, 255, 255]),
        )

        # Find largest component efficiently
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            raise ValueError("No grass found in image")
        largest_component = (
            np.argmax([np.sum(labels == i) for i in range(1, num_labels)]) + 1
        )
        largest_mask = (labels == largest_component).astype(np.uint8) * 255

        # Get mean color directly
        grass_color = np.array(cv2.mean(img, mask=largest_mask)[:3])

        # # TODO: Save masked image if you want to see it
        # # Create masked image efficiently
        # masked_img = cv2.bitwise_and(img, img, mask=largest_mask)

        self.grass_color = grass_color
        return self.grass_color

    def _get_kits_colors(
        self, player_imgs: List[np.ndarray], resize_percent: int = 50
    ) -> List[np.ndarray]:
        """Find the kit colors of all players in the current frame.

        Args:
            players: List of np.array objects that contain the BGR values of the image
            portions that contain players.
            grass_hsv: tuple that contain the HSV color value of the grass color of
            the image background.

        Returns:
            List[np.ndarray]: BGR color values for each player's kit

        Raises:
            ValueError: If grass color has not been extracted or player images are invalid
        """
        if self.grass_color is None:
            raise ValueError("Grass color must be extracted first")
        if not player_imgs:
            raise ValueError("Player images list cannot be empty")

        grass_hsv = cv2.cvtColor(
            np.uint8([[list(self.grass_color)]]), cv2.COLOR_BGR2HSV
        )
        kits_colors = []
        kits_masks = []
        if len(player_imgs) > 200:
            player_imgs_iter = tqdm(player_imgs, desc="Processing players")
        else:
            player_imgs_iter = player_imgs

        for player_img in player_imgs_iter:
            if not isinstance(player_img, np.ndarray):
                raise ValueError("Each player image must be a numpy array")

            # Take the middle 1/2 of the image in both directions
            player_img = player_img[
                player_img.shape[0] // 4 : 3 * player_img.shape[0] // 4,
                player_img.shape[1] // 4 : 3 * player_img.shape[1] // 4,
            ]

            # Downsample the image to reduce the number of pixels
            width = int(player_img.shape[1] * resize_percent / 100)
            height = int(player_img.shape[0] * resize_percent / 100)
            dim = (width, height)

            # Resize image
            player_img = cv2.resize(player_img, dim, interpolation=cv2.INTER_AREA)

            # Convert image to HSV color space
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

            # Define range of green color in HSV
            lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Bitwise-AND mask and original image
            mask = cv2.bitwise_not(mask)
            # add the filtered image as cv2 that is in the mask to the list
            kits_masks.append(cv2.bitwise_and(player_img, player_img, mask=mask))

            # Reshape the masked player image into a list of pixels
            pixels = player_img[mask > 0].reshape(-1, 3)

            # Apply KMeans clustering to find dominant colors
            if len(pixels) > 10:
                # Quantize colors to reduce noise
                # pixels = np.float32(pixels)
                # pixels = pixels // 32 * 32  # Reduce to fewer distinct colors

                # Cluster in HSV space for better color separation
                pixels_hsv = cv2.cvtColor(
                    pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
                ).reshape(-1, 3)
                kmeans = MiniBatchKMeans(
                    n_clusters=3,
                    batch_size=len(pixels_hsv) // 10,
                    n_init="auto",
                    init="k-means++",
                    random_state=46,
                    max_iter=300,  # Increase max iterations
                )
                kmeans.fit(pixels_hsv)

                # Get counts of pixels in each cluster
                labels = kmeans.labels_
                unique_labels, counts = np.unique(labels, return_counts=True)
                # # If clusters are similarly sized (referee case with black/yellow)
                # if (
                #     min(counts) > len(pixels) * 0.3
                # ):  # If secondary cluster is >30% of pixels
                #     print("Secondary cluster is >30% of pixels")
                #     # Choose the brighter cluster
                #     cluster_brightnesses = [
                #         np.mean(pixels_hsv[labels == label, 2])
                #         for label in unique_labels
                #     ]
                #     dominant_cluster = unique_labels[np.argmax(cluster_brightnesses)]
                # else:
                #     # Otherwise use the largest cluster as before
                dominant_cluster = unique_labels[np.argmax(counts)]

                # # Find the most dominant cluster
                # dominant_cluster = unique_labels[np.argmax(counts)]

                # Convert dominant color back to BGR
                dominant_color_hsv = kmeans.cluster_centers_[dominant_cluster].reshape(
                    1, 1, 3
                )
                kit_color = cv2.cvtColor(
                    np.uint8(dominant_color_hsv), cv2.COLOR_HSV2BGR
                ).flatten()
            else:
                # Fallback to mean if no valid pixels or clustering fails
                kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

            kits_colors.append(kit_color)
        return kits_colors

    def _get_kits_classifier(
        self, kits_colors: List[np.ndarray], n_clusters: int = 3
    ) -> KMeans:
        """Create a K-Means classifier for team kit colors.

        Args:
            kits_colors: List of BGR kit colors to train on
            n_clusters: Number of clusters/teams to detect

        Returns:
            KMeans: Trained classifier model

        Raises:
            ValueError: If kit colors are invalid or n_clusters is invalid
        """
        if not kits_colors:
            raise ValueError("Kit colors list cannot be empty")
        if n_clusters < 2:
            raise ValueError("Number of clusters must be at least 2")

        kits_kmeans = KMeans(
            n_clusters=n_clusters, n_init=10, init="k-means++", random_state=40
        )
        kits_kmeans.fit(kits_colors)

        # Get cluster sizes
        labels = kits_kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create mapping from original cluster labels to size-ordered labels
        size_order = np.argsort(-counts)  # Negative for descending order
        label_map = {old: new for new, old in enumerate(size_order)}

        # Store the label mapping in the classifier object
        kits_kmeans.label_map_ = label_map

        # Store original centers in size order
        kits_kmeans.ordered_centers_ = kits_kmeans.cluster_centers_[size_order]

        self.kits_classifier = kits_kmeans
        return kits_kmeans

    def _classify_kits(self, kits_colors: List[np.ndarray]) -> np.ndarray:
        """Classify players into teams based on kit colors.

        Args:
            kits_colors: List of BGR kit colors to classify

        Returns:
            np.ndarray: Team labels (0 or 1) for each kit color

        Raises:
            ValueError: If kit colors are invalid or classifier not trained
        """
        if self.kits_classifier is None:
            raise ValueError("Classifier must be trained first")
        if not kits_colors:
            raise ValueError("Kit colors list cannot be empty")

        # Get raw predictions
        raw_predictions = self.kits_classifier.predict(kits_colors)

        # Map to size-ordered labels
        teams = np.array([self.kits_classifier.label_map_[p] for p in raw_predictions])
        self.teams = teams
        return self.teams

    def _get_left_team_label(self, players_boxes: List[Any], teams: np.ndarray) -> int:
        """Find which team is on the left side of the frame.

        Args:
            players_boxes: Bounding boxes for detected players
            teams: Team classification labels

        Returns:
            int: Label of the team on the left (0 or 1)

        Raises:
            ValueError: If player boxes or team labels are empty/invalid
        """
        logger.info("Getting left team label")
        self.left_team_label = 0
        team_0 = []
        team_1 = []

        if len(players_boxes) == 0:
            raise ValueError("Players boxes list cannot be empty")
        if len(teams) == 0:
            raise ValueError("Teams array cannot be empty")
        if len(players_boxes) != len(teams):
            raise ValueError("Number of player boxes must match number of team labels")

        for i in range(len(players_boxes)):
            x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

            team = teams[i]
            if team == 0:
                team_0.append(np.array([x1]))
            elif team == 1:
                team_1.append(np.array([x1]))
            else:
                continue

        if not team_0 or not team_1:
            raise ValueError("At least one player from each team must be present")

        team_0 = np.array(team_0)
        team_1 = np.array(team_1)

        if np.average(team_0) - np.average(team_1) > 0:
            self.left_team_label = 1

        return self.left_team_label
