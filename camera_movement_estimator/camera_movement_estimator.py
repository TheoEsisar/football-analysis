import pickle
import cv2
import os
import numpy as np
import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    
    def __init__(self, frame):
        """
        Initialize the CameraMovementEstimator object with initial parameters and feature detection settings.

        Parameters:
            frame (numpy.ndarray): Initial video frame used to set up feature detection.
        """
        self.minimum_distance = 5
        
        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1
        
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Estimate camera movement across a sequence of video frames.

        Parameters:
            frames (list): List of video frames to analyze.
            read_from_stub (bool): Flag indicating whether to load camera movement data from a stub file.
            stub_path (str): Path to the stub file for loading/saving camera movement data.

        Returns:
            list: List of estimated camera movements for each frame.
        """
        # Read stub
        if read_from_stub == True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
            return camera_movement
        
        camera_movement = [[0,0]]*len(frames)
        
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        for frame_num in range (1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_feature_point = new.ravel()
                old_feature_point = old.ravel()
                
                distance = measure_distance(new_feature_point, old_feature_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_feature_point, new_feature_point)
                    
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                
            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Annotate video frames with estimated camera movement information.

        Parameters:
            frames (list): List of video frames to annotate.
            camera_movement_per_frame (list): List of camera movements corresponding to each frame.

        Returns:
            list: List of annotated video frames showing camera movement.
        """
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
        
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0), (500,100), (255,255,255), cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)  
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X : {x_movement:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)  
            frame = cv2.putText(frame, f"Camera Movement Y : {y_movement:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)  
            
            output_frames.append(frame)
            
        return output_frames
    
    def add_ajust_position_to_track(self, tracks, camera_movement_per_frame):
        """
        Adjust tracked object positions based on estimated camera movement.

        Parameters:
            tracks (dict): Tracking data for objects across video frames.
            camera_movement_per_frame (list): List of camera movements corresponding to each frame.

        Returns:
            None
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted