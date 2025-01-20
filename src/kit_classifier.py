from src.yolo_model import YoloModel
from collections import defaultdict
import numpy as np
import cv2
from sklearn.cluster import KMeans
import tqdm
from pathlib import Path
from tqdm import tqdm


class KitClassifier:
    def __init__(self):
        """
        Initialize TeamDetector with model and parameters

        Args:
            model_path: Path to YOLO model (if None, loads default)
            n_teams: Number of teams to detect (default 2)
        """
        self.grass_color = None
        self.kits_classifier = None
        self.left_team_label = None

    def extract_grass_color(self, frame):
        """Extract grass color from frame"""
        self.grass_color = self._get_grass_color(frame)
        return self.grass_color

    def get_kit_colors(self, player_imgs):
        """Extract kit colors from cropped player images"""
        if self.grass_color is None:
            raise ValueError("Must extract grass color first")
        return self._get_kits_colors(player_imgs)

    def train_classifier(self, all_kit_colors):
        """Train the classifier on all kit colors"""
        self.kits_classifier = self._get_kits_classifier(all_kit_colors)
        return self.kits_classifier

    def classify_teams(self, kit_colors):
        """Classify kit colors into teams"""
        if self.kits_classifier is None:
            raise ValueError("Must train classifier first")
        return self._classify_kits(kit_colors)

    def determine_left_team(self, players_boxes, team_labels):
        """Determine which team is on the left side"""
        self.left_team_label = self._get_left_team_label(players_boxes, team_labels)
        return self.left_team_label

    def _get_grass_color(self, frame=None):
        """Extract grass color from frame"""
        # Your existing grass color extraction code

        if isinstance(frame, (str, Path)):
            img = cv2.imread(str(frame))
            if img is None:
                raise ValueError(f"Failed to load image from {frame}")
        else:
            # Frame is already a numpy array
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

    def _get_kits_colors(self, player_imgs):
        """
        Finds the kit colors of all the players in the current frame

        Args:
            players: List of np.array objects that contain the BGR values of the image
            portions that contain players.
            grass_hsv: tuple that contain the HSV color value of the grass color of
            the image background.

        Returns:
            kits_colors
                List of np arrays that contain the BGR values of the kits color of all
                the players in the current frame
        """
        if self.grass_color is None:
            self.grass_color = self._get_grass_color()
        grass_hsv = cv2.cvtColor(
            np.uint8([[list(self.grass_color)]]), cv2.COLOR_BGR2HSV
        )
        kits_colors = []
        kits_masks = []
        if len(player_imgs) > 200:  # tmp fix for too many players
            player_imgs_iter = tqdm(player_imgs, desc="Processing players")
        else:
            player_imgs_iter = player_imgs

        for player_img in player_imgs_iter:

            # Take the middle 1/2 of the image in both directions
            player_img = player_img[
                player_img.shape[0] // 4 : 3 * player_img.shape[0] // 4,
                player_img.shape[1] // 4 : 3 * player_img.shape[1] // 4,
            ]

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

                kmeans = KMeans(
                    n_clusters=3,
                    n_init="auto",  # Let sklearn choose optimal number
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
        # self.kits_colors = kits_colors
        return kits_colors

    def _get_kits_classifier(self, kits_colors, n_clusters=3):
        """
        Creates a K-Means classifier that can classify the kits accroding to their BGR
        values into 2 different clusters each of them represents one of the teams

        Args:
            kits_colors: List of np.array objects that contain the BGR values of
            the colors of the kits of the players found in the current frame.

        Returns:
            kits_kmeans
                sklearn.cluster.KMeans object that can classify the players kits into
                2 teams according to their color..
        """
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

    def _classify_kits(self, kits_colors):
        """
        Classifies the player into one of the two teams according to the player's kit
        color

        Args:
            kits_classifer: sklearn.cluster.KMeans object that can classify the
            players kits into 2 teams according to their color.
            kits_colors: List of np.array objects that contain the BGR values of
            the colors of the kits of the players found in the current frame.

        Returns:
            team
                np.array object containing a single integer that carries the player's
                team number (0 or 1)
        """
        # Get raw predictions
        raw_predictions = self.kits_classifier.predict(kits_colors)

        # Map to size-ordered labels
        teams = np.array([self.kits_classifier.label_map_[p] for p in raw_predictions])
        self.teams = teams
        return self.teams

    def _get_left_team_label(self, players_boxes, teams):
        """
        Finds the label of the team that is on the left of the screen

        Args:
            players_boxes: List of ultralytics.engine.results.Boxes objects that
            contain various information about the bounding boxes of the players found
            in the image.
            kits_colors: List of np.array objects that contain the BGR values of
            the colors of the kits of the players found in the current frame.
            kits_clf: sklearn.cluster.KMeans object that can classify the players kits
            into 2 teams according to their color.
        Returns:
            left_team_label
                Int that holds the number of the team that's on the left of the image
                either (0 or 1)
        """
        print("Getting left team label")
        self.left_team_label = 0
        team_0 = []
        team_1 = []

        if len(players_boxes) == 0:
            raise ValueError("Players boxes not found, run _get_players_boxes first")
        if len(teams) == 0:
            raise ValueError("Teams not found, run _classify_kits first")

        for i in range(len(players_boxes)):
            x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

            team = teams[i]
            if team == 0:
                team_0.append(np.array([x1]))
            elif team == 1:
                team_1.append(np.array([x1]))
            else:
                continue

        team_0 = np.array(team_0)
        team_1 = np.array(team_1)

        if np.average(team_0) - np.average(team_1) > 0:
            self.left_team_label = 1

        return self.left_team_label
