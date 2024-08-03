import sys
import cv2
sys.path.append("../")
from utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator():
    
    def __init__(self):
        """
        Initialize the SpeedAndDistanceEstimator object with default settings.

        Sets the frame window size and frame rate for calculating speeds and distances.
        """
        self.frame_window = 5
        self.frame_rate = 21
        
    def add_speed_and_dsitance_to_tracks(self, tracks):
        """
        Calculate and add speed and distance information to the tracking data.

        Calculates the speed and distance covered by players between frames and adds this information to the tracking data.

        Parameters:
            tracks (dict): Tracking data for players, referees, and the ball.

        Returns:
            None
        """
        total_distance = {}
        
        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referee':
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window, number_of_frames-1)
                
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']
                    
                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meters_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meters_per_second*3.6
                    
                    if object not in total_distance:
                        total_distance[object] = {}
                        
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                        
                    total_distance[object][track_id] += distance_covered
                    
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
                        
    def draw_speed_and_distance(self, video_frames, tracks):
        """
        Draw speed and distance annotations onto video frames based on tracking data.

        Parameters:
            video_frames (list): List of video frames to annotate.
            tracks (dict): Tracking data for players, referees, and the ball, including speed and distance information.

        Returns:
            list: List of annotated video frames with speed and distance information.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referee':
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("distance", None)
                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        
                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f"{distance:.2f} meters", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        
            output_video_frames.append(frame)
                    
        return output_video_frames
                    
                        