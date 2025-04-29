# STEPP/utils/camera.py
class Camera:
    def __init__(self, width, height, camera_name, distortion_model, K, D, Rect, P):
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.distortion_model = distortion_model
        self.K = K  # Intrinsic matrix
        self.D = D  # Distortion coefficients
        self.Rect = Rect  # Rectification matrix
        self.P = P  # Projection matrix

    def project(self, point):
        """
        Project a 3D point into the 2D image plane.
        Args:
            point: 3D point (numpy array of shape (3,))
        Returns:
            pixel coordinates (u, v)
        """
        # This is a placeholder; CameraPinhole overrides it with cv2.projectPoints
        raise NotImplementedError("Subclasses must implement project()")