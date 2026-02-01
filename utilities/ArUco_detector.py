import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from collections import Counter, defaultdict

from .camera import Camera

def get_projection_matrices(cameras: list[Camera]):
    proj_matrices = []
    for cam in cameras:
        K = cam.intrinsic
        extrinsic = cam.extrinsic_matrix

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3].reshape(3, 1)

        Rt = np.hstack((R, t))  # shape (3, 4)
        P = K @ Rt  # projection matrix: shape (3, 4)
        proj_matrices.append(P)
    return proj_matrices


def undistort_corners(corners, camera: Camera):
    undistorted_corners = cv2.undistortPoints(
        corners.reshape(-1, 1, 2),
        cameraMatrix=camera.intrinsic,
        distCoeffs=camera.distortion_coeff,
        P=camera.intrinsic,  # reprojects into pixel coords
    ).reshape(-1, 2)
    return undistorted_corners


def triangulate_n_views(image_points, projection_matrices):
    A = []
    for x, P in zip(image_points, projection_matrices):
        x, y = x[0], x[1]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def reproject(P, X):
    X_h = np.append(X, 1.0)
    proj = P @ X_h
    return proj[:2] / proj[2]

def get_marker_positions(
    detections,
    cameras: list,
    input_is_undistorted=True,
):
    num_cameras = len(cameras)
    num_frames = len(detections[0])
    proj_matrices = [cam.intrinsic @ cam.extrinsic_matrix[:3, :] for cam in cameras]

    # Structure: {frame_idx: {marker_id: [x, y, z]}}
    all_marker_positions = {}

    for frame in range(num_frames):
        image_points_per_cam = []
        detected_ids_per_cam = []

        # 1. Collect and (optionally) undistort points for this frame
        for cam_idx in range(num_cameras):
            detection = detections[cam_idx][frame]
            ids = detection.get("ids")
            corners = detection.get("corners")

            if ids is None or corners is None or len(ids) == 0:
                image_points_per_cam.append({})
                detected_ids_per_cam.append(set())
                continue

            ids = ids.flatten()
            corners = np.array(corners, dtype=np.float32).reshape(-1, 2)

            if not input_is_undistorted:
                # Assuming undistort_corners is defined in your environment
                corners = undistort_corners(corners, cameras[cam_idx])

            point_map = {int(id_): corners[i] for i, id_ in enumerate(ids)}
            image_points_per_cam.append(point_map)
            detected_ids_per_cam.append(set(point_map.keys()))

        # 2. Find IDs visible in at least 2 cameras
        all_ids_flat = [id_ for ids_set in detected_ids_per_cam for id_ in ids_set]
        id_counts = Counter(all_ids_flat)
        common_ids = {id_ for id_, count in id_counts.items() if count >= 2}

        frame_results = {}

        # 3. Triangulate
        for point_id in common_ids:
            views = []
            Ps = []

            for cam_idx in range(num_cameras):
                if point_id in image_points_per_cam[cam_idx]:
                    views.append(image_points_per_cam[cam_idx][point_id])
                    Ps.append(proj_matrices[cam_idx])

            # Triangulate into 3D world coordinates
            pt_3d = triangulate_n_views(views, Ps)
            frame_results[point_id] = pt_3d

        all_marker_positions[frame] = frame_results

    return all_marker_positions

def visualize_detections_with_cross(image_rgb, detection, title="ArUco Detections"):
    # Create a copy to keep the original clean
    vis_img = image_rgb.copy()
    
    ids = detection.get("ids")
    corners = detection.get("corners")

    if ids is not None and len(ids) > 0:
        # Drawing colors (BGR format for OpenCV)
        blue_color = (255, 0, 0) # Pure Blue in RGB (Matplotlib will see it correctly)
        thickness = 15           # Adjust for "Big" cross appearance
        cross_size = 60          # Half-length of the cross arms

        for marker_corners in corners:
            # marker_corners shape is usually (1, 4, 2)
            pts = marker_corners.reshape(4, 2)
            
            # Calculate the center of the marker
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            # Draw the two diagonal lines for a cross
            # Line 1: Top-left to bottom-right
            cv2.line(vis_img, 
                     (center_x - cross_size, center_y - cross_size), 
                     (center_x + cross_size, center_y + cross_size), 
                     blue_color, thickness)
            
            # Line 2: Top-right to bottom-left
            cv2.line(vis_img, 
                     (center_x + cross_size, center_y - cross_size), 
                     (center_x - cross_size, center_y + cross_size), 
                     blue_color, thickness)

        print(f"Found {len(ids)} markers. Visualizing with blue crosses.")
    else:
        print("No markers detected.")

    plt.figure(figsize=(12, 8))
    plt.imshow(vis_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

    return vis_img

def add_markers_markers_to_fig(fig, marker_data):
    # 1. Process Marker Data
    # We collect all points into a single (N, 3) numpy array for speed
    all_points = []
    marker_labels = []
    
    for frame_idx, markers in marker_data.items():
        for marker_id, pos in markers.items():
            all_points.append(pos)
            marker_labels.append(f"F{frame_idx}_ID{marker_id}")
    
    if all_points:
        pts_np = np.array(all_points) # Shape: (N, 3)
        
        fig.add_trace(go.Scatter3d(
            x=pts_np[:, 0],
            y=pts_np[:, 1],
            z=pts_np[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='blue', opacity=0.8),
            text=marker_labels,
            name='Markers'
        ))

    return fig

def add_cameras_to_fig(fig, cameras, scale=0.2): 
    # Lists for frustum lines
    all_x, all_y, all_z = [], [], []
    
    # Lists for camera centers and labels
    cam_centers = []
    cam_labels = []

    for cam in cameras:
        if cam.extrinsic_matrix is None:
            continue
            
        # The Camera-to-World transformation
        # R_cw = R^T, t_cw = -R^T @ t
        R_cw = cam.rot.T
        t_cw = cam.world_pos # Uses your property: -self.rot.T @ self.t
        
        cam_centers.append(t_cw)
        cam_labels.append(cam.name if cam.name else "Unknown")
        
        # --- Frustum Calculation ---
        # Define pyramid corners in local camera space (Z is forward)
        s = scale
        corners_cam = np.array([
            [0, 0, 0],          # 0: Camera Center
            [-s, -s, s*1.5],    # 1: Top-Left
            [s, -s, s*1.5],     # 2: Top-Right
            [s, s, s*1.5],      # 3: Bottom-Right
            [-s, s, s*1.5],     # 4: Bottom-Left
        ])
        
        # Transform corners to World Space: P_world = R_cw @ P_cam + t_cw
        p = (R_cw @ corners_cam.T).T + t_cw
        
        # Line sequence to draw the wireframe pyramid
        indices = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 2, 3, 4, 1]
        for idx in indices:
            all_x.append(p[idx, 0]); all_y.append(p[idx, 1]); all_z.append(p[idx, 2])
        # Add None to break the line between different cameras
        all_x.append(None); all_y.append(None); all_z.append(None)

    # 1. Add Frustums Trace
    fig.add_trace(go.Scatter3d(
        x=all_x, y=all_y, z=all_z,
        mode='lines',
        line=dict(color='rgba(100, 100, 255, 0.6)', width=2),
        name='Frustums',
        hoverinfo='none'
    ))

    # 2. Add Camera Centers Trace
    if cam_centers:
        cam_centers = np.array(cam_centers)
        fig.add_trace(go.Scatter3d(
            x=cam_centers[:, 0],
            y=cam_centers[:, 1],
            z=cam_centers[:, 2],
            mode='markers+text',
            marker=dict(
                size=5,
                color=np.arange(len(cam_centers)), 
                colorscale='Viridis',
                opacity=0.9
            ),
            text=cam_labels,
            textposition="top center",
            name='Camera Positions'
        ))

    # 3. Add World Origin (Stage Center)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='red', symbol='cross'),
        name='Stage Center'
    ))
    return fig