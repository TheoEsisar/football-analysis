import sys
sys.path.append("../")
from utils import get_center_of_box, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        """
        Initialize the PlayerBallAssigner object with a default maximum distance between a player and the ball.

        Attributes:
            max_player_ball_distance (float): Maximum distance allowed between a player and the ball for assignment.
        """
        self.max_player_ball_distance = 70
        
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assign the ball to the closest player based on the distance between the player and the ball.

        Parameters:
            players (dict): Dictionary of players where keys are player IDs and values are dictionaries containing player information including bounding boxes.
            ball_bbox (list): Bounding box of the ball.

        Returns:
            int: ID of the player assigned the ball, or -1 if no player is close enough.
        """
        ball_position = get_center_of_box(ball_bbox)
        
        minimum_distance = 99999
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player["bbox"]
            
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)
            
            if distance<self.max_player_ball_distance:
                if distance<minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
                    
        return assigned_player