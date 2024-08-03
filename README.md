# Football Analysis Using Computer Vision

This Python project focuses on analyzing football (soccer) matches using computer vision techniques. The project aims to track players and the ball, estimate camera movements, transform views, calculate speeds and distances, assign players to teams, and determine ball control throughout the match.

## Features

- Player and Ball Tracking: Utilizes YOLOv8x model made for football players detection to follow the movements of players and the ball across video frames.
- Camera Movement Estimation: Estimates the movement of the camera to adjust player and ball positions accordingly.
- View Transformation: Transforms the perspective of the video.
- Speed and Distance Calculation: Calculates the speed of players and the distance they cover during the match.
- Team Assignment: Identifies and assigns players to their respective teams based on jersey colors.
- Ball Control Determination: Determines which player has control of the ball at any given time and assigns ball possession to teams.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project requires several external libraries such as OpenCV, NumPy, and PyTorch. You can install these dependencies using pip:

```sh
pip install opencv-python numpy scikit-learn ultralytics pandas
```

### Installation

Clone this repository to your local machine:

```sh
git clone https://github.com/TheoEsisar/football-analysis.git
cd football-analysis
```

### Usage

To analyze a football video, simply run the main.py script with the path to your input video file:

```sh
python main.py
```

The script processes the video and generates an output video with annotations showing player movements, ball possession, camera movement adjustments, speed and distance calculations, and team assignments.
