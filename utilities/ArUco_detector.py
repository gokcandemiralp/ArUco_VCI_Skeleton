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

def visualize_markers_3d_plotly(marker_data, cameras=None):
    fig = go.Figure()

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

    # 2. Add Camera Positions from the Camera Class
    if cameras:
        cam_positions = []
        cam_names = []
        
        for cam in cameras:
            pos = cam.world_pos  #
            if pos is not None:
                cam_positions.append(pos)
                cam_names.append(cam.name)
        
        if cam_positions:
            cams_np = np.array(cam_positions)
            
            fig.add_trace(go.Scatter3d(
                x=cams_np[:, 0],
                y=cams_np[:, 1],
                z=cams_np[:, 2],
                mode='markers+text',
                marker=dict(size=4, color='black', symbol='diamond'),
                text=cam_names,
                name='Cameras'
            ))

    # 3. Scene Layout & Calibration Alignment
    # The dome Z-axis is forward into the scene
    fig.update_layout(
        title="Dome ArUco 3D Locations (Numpy)",
        scene=dict(
            aspectmode='data',  # Keeps 1:1:1 scale proportions
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig