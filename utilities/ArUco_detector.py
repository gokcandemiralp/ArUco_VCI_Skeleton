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
            point_map = {}

            # Iterate over the markers properly
            for i, marker_id in enumerate(ids):
                # Extract the 4 corners for THIS specific marker
                # corners[i] gives us the shape (1, 4, 2) or (4, 2)
                marker_corners = np.array(corners[i]).reshape(-1, 2)

                # OPTIONAL: Undistort specific points here if needed
                if not input_is_undistorted:
                     # Make sure this returns PIXELS, not normalized coords
                     marker_corners = undistort_corners(marker_corners, cameras[cam_idx])

                # Calculate the CENTER of the marker (average of 4 corners)
                # This gives a stable single point representing the marker center
                marker_center = np.mean(marker_corners, axis=0)
                
                point_map[int(marker_id)] = marker_center

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

def extract_2d_marker_positions(detection):
    marker_data = []
    ids = detection.get("ids")
    corners = detection.get("corners")

    if ids is not None and len(ids) > 0:
        ids = ids.flatten()
        for i, marker_corners in enumerate(corners):
            # Reshape to (4, 2) to get the four corners
            pts = marker_corners.reshape(4, 2)
            
            # Calculate the centroid (arithmetic mean of corners)
            center_x = float(np.mean(pts[:, 0]))
            center_y = float(np.mean(pts[:, 1]))
            
            marker_data.append({
                "id": int(ids[i]),
                "center": (center_x, center_y)
            })
            
    return marker_data

def get_all_camera_2d_marker_positions(multi_camera_detections):
    results = {}
    
    for cam_id, detection in multi_camera_detections.items():
        results[cam_id] = extract_2d_marker_positions(detection)
        
    return results

def visualize_detections_on_image(image_rgb, detection, title="ArUco Detections"):
    vis_img = image_rgb.copy()
    
    # Use the helper to get clean data
    markers = extract_2d_marker_positions(detection)

    if markers:
        # --- Visual Settings ---
        circle_color = (255, 0, 0)         # Blue
        text_color_inner = (0, 0, 0)       # Black
        text_color_outer = (255, 255, 255) # White
        
        radius = 60
        thickness_circle = 10
        font_scale = 3.6
        font_thickness = 12
        outline_thickness = 25

        for marker in markers:
            m_id = marker["id"]
            center_x, center_y = map(int, marker["center"])

            # 1. Draw the Circle
            cv2.circle(vis_img, (center_x, center_y), radius, circle_color, thickness_circle)
            
            # Text Position logic
            text_pos = (center_x + radius + 10, center_y + 10)
            text_str = str(m_id)

            # 2. Draw the High-Contrast Text (Outer then Inner)
            cv2.putText(vis_img, text_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, text_color_outer, outline_thickness)
            cv2.putText(vis_img, text_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, text_color_inner, font_thickness)

        print(f"Found {len(markers)} markers. Visualizing detections.")
    else:
        print("No markers detected.")

    plt.figure(figsize=(12, 8))
    plt.imshow(vis_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def add_markers_to_fig(fig, marker_data):
    # 1. Process Marker Data
    # We collect all points into a single (N, 3) numpy array for speed
    all_points = []
    marker_labels = []
    
    for frame_idx, markers in marker_data.items():
        for marker_id, pos in markers.items():
            all_points.append(pos)
            marker_labels.append(f"M{marker_id}")
    
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

def add_skeleton_to_fig(fig, marker_data, marker_pairs):
    line_x = []
    line_y = []
    line_z = []

    for frame_idx, markers in marker_data.items():
        for (m_start, m_end) in marker_pairs:
            # Check if both markers exist in the current frame
            if m_start in markers and m_end in markers:
                p1 = markers[m_start]
                p2 = markers[m_end]

                # Append segment: Start -> End -> None (to break the line)
                line_x.extend([p1[0], p2[0], None])
                line_y.extend([p1[1], p2[1], None])
                line_z.extend([p1[2], p2[2], None])

    if line_x:
        fig.add_trace(go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode='lines',
            line=dict(color='red', width=3),
            name='Connections',
            connectgaps=False # Ensure segments stay separated
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