from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from utils import get_center_of_box, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        """
        Initialize the Tracker object with a given model path for object detection.

        Parameters:
            model_path (str): Path to the pre-trained model for object detection.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def interpolate_ball_position(self, ball_positions):
        """
        Interpolate missing ball positions using pandas DataFrame interpolation methods.

        Parameters:
            ball_positions (list): A list of dictionaries containing ball position data.

        Returns:
            list: A list of interpolated ball positions.
        """
        ball_positions = [x.get(1, {}).get("bbox", [])for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions

    def detect_frames(self, frames):
        """
        Detect objects in a batch of video frames using the YOLO model.

        Parameters:
            frames (list): A list of video frames to detect objects in.

        Returns:
            list: A list of detections for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections+=detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track objects across video frames and optionally load/save tracking data from/to a stub file.

        Parameters:
            frames (list): A list of video frames to track objects in.
            read_from_stub (bool): Flag indicating whether to load tracking data from a stub file.
            stub_path (str): Path to the stub file for loading/saving tracking data.

        Returns:
            dict: A dictionary containing tracking data for players, referees, and the ball.
        """
        
        if read_from_stub == True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # Convert detection to supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]
                    
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse around an object (player/referee) in a video frame.

        Parameters:
            frame (numpy.ndarray): Video frame to draw on.
            bbox (list): Bounding box coordinates of the object.
            color (tuple): Color of the ellipse.
            track_id (int, optional): Tracking ID of the object. Defaults to None.

        Returns:
            numpy.ndarray: Video frame with the ellipse drawn around the object.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_box(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(frame, (x_center, y2), (int(width), int(width*0.35)), angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15
        
        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            
            x1_text = x1_rect+6
            if track_id<10:
                x1_text += 6
            elif track_id>99:
                x1_text -= 6
                
            cv2.putText(frame, f"{track_id}", (int(x1_text), y1_rect+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return frame
    
    def draw_triangle(self, frame,bbox,color):
        """
        Draw a triangle above an object (ball) in a video frame.

        Parameters:
            frame (numpy.ndarray): Video frame to draw on.
            bbox (list): Bounding box coordinates of the object.
            color (tuple): Color of the triangle.

        Returns:
            numpy.ndarray: Video frame with the triangle drawn above the object.
        """
        y=int(bbox[1])
        x, _ = get_center_of_box(bbox)
        
        triangle_points = np.array([[x,y], [x-10, y-20], [x+10, y-20]])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        
        return frame
        
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw a visualization of team ball control over time on a video frame.

        Parameters:
            frame (numpy.ndarray): Video frame to draw on.
            frame_num (int): Current frame number.
            team_ball_control (numpy.ndarray): Array indicating which team had ball control in each frame.

        Returns:
            numpy.ndarray: Video frame with the team ball control visualization.
        """
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        ball_control_team_1 = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        ball_control_team_2 = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        team1 = ball_control_team_1/(ball_control_team_1+ball_control_team_2)
        team2 = ball_control_team_2/(ball_control_team_1+ball_control_team_2)
        
        cv2.putText(frame, f"Team 1 Ball Control : {team1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control : {team2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        
        return frame
        
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Annotate video frames with tracking data and visualizations.

        Parameters:
            video_frames (list): List of video frames to annotate.
            tracks (dict): Tracking data for players, referees, and the ball.
            team_ball_control (numpy.ndarray): Array indicating which team had ball control in each frame.

        Returns:
            list: List of annotated video frames.
        """
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                
                if player.get('has_ball', False):
                    self.draw_triangle(frame, player['bbox'], (0,0,255))
                    
            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255))
                
            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))
                
            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)    
            
            output_frames.append(frame)
            
        return output_frames
    
    def add_position_to_track(self, tracks):
        """
        Add position information to tracking data for each tracked object.

        Parameters:
            tracks (dict): Tracking data for players, referees, and the ball.

        Returns:
            None
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_box(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position
                