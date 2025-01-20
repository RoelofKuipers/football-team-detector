from ultralytics import YOLO


class YoloModel:
    """Handles all YOLO model loading and object detection"""

    def __init__(self, model_path=None, class_names=None):
        self.model = self._load_model(model_path)
        self.class_names = class_names
        self.classes_to_keep = self._get_classes_to_keep()

    def _load_model(self, model_path):
        if not model_path:
            raise ValueError("Model path is required")
        return YOLO(model_path)

    def _get_classes_to_keep(self):
        if not self.class_names:
            return list(self.model.names.keys())
        return [k for k, v in self.model.names.items() if v in self.class_names]

    def detect(self, frame):
        """Run detection on a single frame"""
        results = self.model.predict(
            source=frame,
            conf=0.3,
            iou=0.3,
            classes=self.classes_to_keep,
            verbose=False,
        )
        return results[0]
