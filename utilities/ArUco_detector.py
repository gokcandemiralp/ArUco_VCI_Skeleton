import os
import cv2
import numpy as np
import math
import json
from collections import Counter
import re

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

def detections_to_json(marker_positions, output_path):
    json_data = {}
    for frame_idx, markers in marker_positions.items():
        frame_key = str(frame_idx)
        json_data[frame_key] = {}
        for marker_id, position_array in markers.items():
            json_data[frame_key][str(marker_id)] = position_array.tolist()

    # 2. Convert to JSON string with standard indentation first
    json_str = json.dumps(json_data, indent=4)

    # 3. Use Regex to collapse lists [ ... ] onto a single line
    # Matches brackets with newlines inside and removes the whitespace/newlines
    json_str = re.sub(
        r'\[\s+([^]]+)\s+\]',  # Pattern: [ followed by content and whitespace ]
        lambda m: '[' + ', '.join([x.strip() for x in m.group(1).split(',')]) + ']',
        json_str
    )

    # 4. Ensure directory exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # 5. Write the formatted string to file
    with open(output_path, 'w') as f:
        f.write(json_str)

    print(f"Saved formatted marker positions to {output_path}")

    import numpy as np

def clean_invalid_markers(corners, ids, valid_range=(0, 11)):
    if ids is None:
        return corners, ids

    # Create a boolean mask for valid IDs (ids is shape (N, 1))
    # We flatten it to shape (N,) for simple boolean comparison
    mask = (ids.flatten() >= valid_range[0]) & (ids.flatten() <= valid_range[1])

    # If all are valid, return as is
    if np.all(mask):
        return corners, ids

    # If none are valid, return empty/None structure
    if not np.any(mask):
        return (), None

    # Filter IDs using the mask
    cleaned_ids = ids[mask]

    # Filter Corners (corners is a tuple, so we use zip with the mask)
    cleaned_corners = tuple([c for c, is_valid in zip(corners, mask) if is_valid])

    return cleaned_corners, cleaned_ids

# Fix the idx's for ids to correspond exactly so for example the joint indext 0 must correspond to right hand wrist for both of them
# Change the coordinate scaling, convert everything to meters 
# The predictions have a completely different coordinate system, swap the y and z coordinates and negate the whole coordinate
def convert_predictions(pred_data, mapping):
    processed_predictions = {}
    
    for frame_id, joints in pred_data.items():
        processed_predictions[frame_id] = {}
        
        for pred_idx, gt_idx in mapping:
            # JSON keys are strings, mapping provides ints
            pred_key = str(pred_idx)
            gt_key = str(gt_idx)
            
            if pred_key in joints:
                raw_coords = joints[pred_key]
                x_mm, y_mm, z_mm = raw_coords
                
                # Scale to meters
                x_m = x_mm / 1000.0
                y_m = y_mm / 1000.0
                z_m = z_mm / 1000.0
                
                # Transform coordinate system
                # "Swap y and z coordinates and negate the whole coordinate"
                # Input: (x, y, z)
                # Swap Y/Z: (x, z, y)
                # Negate: (-x, -z, -y)
                new_x = -x_m
                new_y = -z_m
                new_z = -y_m
                
                processed_predictions[frame_id][gt_key] = [new_x, new_y, new_z]
                
    return processed_predictions

def compare_sequences(ground_truth, processed_preds):
    frame_diffs = {}
    
    # Iterate over frames present in Ground Truth
    for frame_id, gt_joints in ground_truth.items():
        if frame_id not in processed_preds:
            continue
            
        pred_joints = processed_preds[frame_id]
        frame_diffs[frame_id] = {}
        
        # Iterate over joints present in Ground Truth frame
        for joint_id, gt_coord in gt_joints.items():
            if joint_id in pred_joints:
                pred_coord = pred_joints[joint_id]
                
                # Calculate Euclidean distance
                dist = math.sqrt(
                    (gt_coord[0] - pred_coord[0])**2 +
                    (gt_coord[1] - pred_coord[1])**2 +
                    (gt_coord[2] - pred_coord[2])**2
                )
                
                frame_diffs[frame_id][joint_id] = dist
                
    return frame_diffs