import numpy as np
import json

class Camera:
    extrinsic_matrix = None
    name = None

    def __init__(self, intrinsic, distortion_coeff, image_size):
        self.intrinsic = intrinsic
        self.distortion_coeff = distortion_coeff
        self.image_size = image_size

    @property
    def rot(self) -> np.ndarray:
        if self.extrinsic_matrix is None:
            return None
        else:
            return self.extrinsic_matrix[:3, :3]  # R

    @property
    def t(self) -> np.ndarray:
        if self.extrinsic_matrix is None:
            return None
        else:
            return self.extrinsic_matrix[:3, 3]  # T

    @property
    def world_pos(self) -> np.ndarray:
        if self.extrinsic_matrix is None:
            return None
        else:
            return -self.rot.T @ self.t
        
    def adjust_intrinsic_for_scale(self, scale_factor):
        # Create a copy to avoid modifying the original data
        K_scaled = self.intrinsic.copy()
        
        # Scale fx and fy (focal lengths)
        K_scaled[0, 0] *= scale_factor
        K_scaled[1, 1] *= scale_factor
        
        # Scale cx and cy (principal point / optical center)
        K_scaled[0, 2] *= scale_factor
        K_scaled[1, 2] *= scale_factor
        
        # If there is a skew factor (K[0,1]), it must also be scaled
        K_scaled[0, 1] *= scale_factor
        self.intrinsic = K_scaled
        if self.image_size is not None:
            self.image_size = (
                int(self.image_size[0] * scale_factor),
                int(self.image_size[1] * scale_factor)
            )

        return K_scaled

def load_cameras_from_json(json_path, camera_names_to_load):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    loaded_cameras = []
    
    # Create a lookup for camera IDs to avoid nested loops
    camera_lookup = {cam['camera_id']: cam for cam in data['cameras']}
    
    for name in camera_names_to_load:
        if name not in camera_lookup:
            print(f"Warning: Camera {name} not found in calibration file.")
            continue
            
        cam_data = camera_lookup[name]
        
        # 1. Parse Intrinsics (reshape flat 9-element list to 3x3)
        intrinsic_matrix = np.array(cam_data['intrinsics']['camera_matrix']).reshape(3, 3)
        
        # 2. Parse Distortion Coefficients
        dist_coeffs = np.array(cam_data['intrinsics']['distortion_coefficients'])
        
        # 3. Parse Image Size/Resolution
        image_size = tuple(cam_data['intrinsics']['resolution'])
        
        # 4. Initialize Camera Object
        cam_obj = Camera(intrinsic_matrix, dist_coeffs, image_size)
        cam_obj.name = name
        
        # 5. Parse Extrinsics (reshape flat 16-element list to 4x4)
        # Note: The JSON uses row_major layout as specified in meta.calibration.matrix_layout
        extrinsic_matrix = np.array(cam_data['extrinsics']['view_matrix']).reshape(4, 4)
        cam_obj.extrinsic_matrix = extrinsic_matrix
        
        loaded_cameras.append(cam_obj)
        
    return loaded_cameras