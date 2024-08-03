from sklearn.cluster import KMeans

class TeamAssigner:
    
    def __init__(self):
        """
        Initialize the TeamAssigner object with empty dictionaries for team colors and player-team assignments.
        """
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        """
        Perform K-means clustering on the reshaped input image to identify dominant colors.

        Parameters:
            image (numpy.ndarray): Input image to perform clustering on.

        Returns:
            sklearn.cluster.KMeans: Fitted KMeans clustering model.
        """
        # Reshape image to 2D array
        image_2d = image.reshape(-1,3)
        
        # K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        
        return kmeans
        
    def get_player_color(self, frame, bbox):
        """
        Determine the dominant color of a player in a given frame based on bounding box coordinates.

        Parameters:
            frame (numpy.ndarray): Frame containing the player.
            bbox (list): Bounding box coordinates of the player.

        Returns:
            numpy.ndarray: Dominant color of the player as determined by K-means clustering.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]
        
        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
        # Get cluster labels for each pixels
        labels = kmeans.labels_
        
        # Reshape labels to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Get player cluster
        corner_cluster = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        """
        Assign team colors based on the dominant colors of detected players in a frame.

        Parameters:
            frame (numpy.ndarray): Frame containing players.
            player_detections (dict): Dictionary of player detections with bounding boxes.
        """
        player_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
            
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
    def get_player_team(self, frame, player_bbox, player_id):
        """
        Assign a team ID to a player based on their dominant color.

        Parameters:
            frame (numpy.ndarray): Frame containing the player.
            player_bbox (list): Bounding box coordinates of the player.
            player_id (int): Unique identifier for the player.

        Returns:
            int: Team ID assigned to the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)
        
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id