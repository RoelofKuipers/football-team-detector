# Football Team Detection and Classification

This project detects football players in video footage and classifies them into teams based on their kit colors.

## Features
- Player detection using YOLOv8
- Team classification based on kit colors
- Referee detection
- Output video with team annotations
- Frame-by-frame analysis

## Directory Structure
```
project/
├── data/
│ └── sample_video.mp4
├── output/
│ ├── frames/ # Annotated frames
│ ├── match_processed.mp4
│ └── teams_per_frame.json
├── checkpoints/
│ └── yolo_football.pt # Custom YOLO weights
├── src/
│ ├── __init__.py
│ ├── yolo_model.py
│ ├── kit_classifier.py
│ ├── player_tracker.py
│ ├── video_processor.py
│ └── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```
Tests still to be added

## Installation

### Using Docker 

```
# Build the image
docker build -t football-team-detector .
 
# Run with video from data directory
docker run --rm -v $(pwd)/data:/data football-team-detector \
    python main.py -i /data/sample.mp4
```

### Manual Installation
Create virtual environment
` conda create -n football-detector python==3.11`

Install requirements
`pip install -r requirements.txt`



## Usage

### Command Line

```
python main.py -i path/to/video.mp4
```

### API

[TO DO]

## Requirements
- Python 3.11
- OpenCV
- YOLOv8
- See requirements.txt for full list

## Output
The system generates:
1. Annotated video with team classifications
2. Frame-by-frame JSON analysis
3. Extracted frames with team annotations


### Detection Visualization
The output video and frames include:
- Red boxes: Team 1 players
- Blue boxes: Team 2 players
- Green boxes: Referees
- Player count display for each team
- Labels indicating player roles and team assignments

Example visualization:
![Detection Example](docs/detection_example.jpg)

## Model

### Required YOLO Classes
The model requires a pre-trained YOLO model that can detect these specific classes:
- "Player"
- "Main Referee"
- "Side Referee"
- "GoalKeeper"

### Model Weights
This project uses specific pre-trained weights from the SoccerNet dataset. You can obtain the weights in two ways:

1. **Download from Shared Link**:
   - Download the weights from: `[LINK still needed]`
   - Place the weights file in: `checkpoints/yolo_football.pt`

### Important Note
Custom training is not currently supported as it requires specific SoccerNet dataset access and preprocessing. Please use the provided pre-trained weights.
