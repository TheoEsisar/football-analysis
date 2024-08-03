import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        """
        Initialize the ViewTransformer object with predefined court dimensions and vertices for perspective transformation.

        Parameters:
            None

        Initializes the transformer with court dimensions and calculates the perspective transform matrix.
        """
        court_width = 68
        court_length = 23.32
        
        self.pixel_verticies = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])
        
        self.target_verticies = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])
        
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)
        
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)
        
    def transform_point(self, point):
        """
        Transform a point from the original frame to the transformed view using perspective transformation.

        Parameters:
            point (numpy.ndarray): Point to be transformed, represented as a NumPy array [x, y].

        Returns:
            numpy.ndarray: Transformed point coordinates as a NumPy array [x_transformed, y_transformed].
                           Returns None if the point is outside the defined polygon.
        """
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >=0
        if not is_inside:
            return None
        
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point,  self.perspective_transformer)
        transformed_point = transformed_point.reshape(-1,2)
        
        return transformed_point
        
        
    def add_transformed_position_to_tracks(self, tracks):
        """
        Add transformed position information to tracking data for each tracked object.

        Parameters:
            tracks (dict): Tracking data for objects including their adjusted positions.

        Modifies the input dictionary in place to include transformed positions for each object in each frame.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]["position_transformed"] = position_transformed
                    