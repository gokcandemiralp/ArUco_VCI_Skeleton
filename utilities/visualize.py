import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import numpy as np
import cv2

from utilities.ArUco_detector import extract_2d_marker_positions

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

def create_error_timeline_gif(differences, output_path="error_timeline.gif", duration_ms=100):
    # --- 1. Prepare the data ---
    # Sort frames numerically
    sorted_frame_ids = sorted([int(k) for k in differences.keys()])
    
    frames = []
    avg_errors = []
    
    for fid in sorted_frame_ids:
        frames.append(fid) # Always add the frame ID to keep x-axis continuous
        
        fid_str = str(fid)
        diff_data = differences.get(fid_str, {})
        
        if diff_data:
            # Calculate average if data exists
            avg_error = sum(diff_data.values()) / len(diff_data)
            avg_errors.append(avg_error)
        else:
            # Append None to create a gap in the line plot
            avg_errors.append(None)
    
    # Filter out None values just to calculate axis limits safely
    valid_errors = [e for e in avg_errors if e is not None]
    
    if not valid_errors:
        print("No valid data points to plot.")
        return

    # --- 2. Setup the Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the line (Matplotlib will break the line where data is None)
    line, = ax.plot(frames, avg_errors, color='royalblue', linewidth=2, label='Avg Error (m)')
    
    # Styling
    ax.set_title(f'Average Euclidean Error per Frame', fontsize=14)
    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('Error (meters)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set axis limits
    ax.set_xlim(min(frames), max(frames))
    y_max = max(valid_errors)
    ax.set_ylim(0, y_max * 1.1) 

    # --- 3. Initialize Animation Elements ---
    # Vertical cursor line
    cursor = ax.axvline(x=frames[0], color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Text annotation box
    text_annotation = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right')

    # --- 4. Define Update Function ---
    def update(frame_idx):
        # frame_idx is the index in our lists (0 to len(frames)-1)
        current_fid = frames[frame_idx]
        current_error = avg_errors[frame_idx]
        
        # Move the red cursor
        cursor.set_xdata([current_fid])
        
        # Update text
        if current_error is not None:
            status_text = f"Frame: {current_fid}\nError: {current_error:.4f} m"
        else:
            status_text = f"Frame: {current_fid}\nError: No Data"
            
        text_annotation.set_text(status_text)
        
        return cursor, text_annotation

    # --- 5. Create and Save Animation ---
    print(f"Generating GIF with {len(frames)} frames...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames), # Iterate over the number of indices
        interval=duration_ms, 
        blit=True
    )
    
    ani.save(output_path, writer='pillow')
    print(f"Saved to {output_path}")
    plt.close()

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

def create_speed_timeline_gif(predictions, output_path="speed_timeline.gif", duration_ms=100, fps=30, title = 'Average Joint Speed per Frame', line_color='darkorange'):
    # --- 1. Prepare the data ---
    # Sort frames numerically
    sorted_frame_ids = sorted([int(k) for k in predictions.keys()])
    
    frames = []
    avg_speeds = []
    
    for i in range(len(sorted_frame_ids)):
        curr_fid = sorted_frame_ids[i]
        frames.append(curr_fid)
        
        curr_fid_str = str(curr_fid)
        curr_data = predictions.get(curr_fid_str, {})
        
        # If it's the first frame or the current frame is empty, we can't calculate speed
        if i == 0 or not curr_data:
            avg_speeds.append(None)
            continue
            
        # Get the previous frame to calculate the difference
        prev_fid = sorted_frame_ids[i-1]
        prev_data = predictions.get(str(prev_fid), {})
        
        if not prev_data:
            avg_speeds.append(None)
            continue
            
        # Time difference in seconds (accounts for potential gaps in frame IDs)
        frame_diff = curr_fid - prev_fid
        time_diff = frame_diff / fps
        
        joint_speeds = []
        for joint_id, curr_coords in curr_data.items():
            if joint_id in prev_data:
                prev_coords = prev_data[joint_id]
                
                # Calculate Euclidean distance between the joint's previous and current position
                dist = math.sqrt(sum((c - p) ** 2 for c, p in zip(curr_coords, prev_coords)))
                
                # Speed = distance / time (m/s)
                speed = dist / time_diff
                joint_speeds.append(speed)
        
        # Average the speeds of all valid joints for this frame
        if joint_speeds:
            avg_speeds.append(sum(joint_speeds) / len(joint_speeds))
        else:
            avg_speeds.append(None)

    # Filter out None values to calculate axis limits safely
    valid_speeds = [s for s in avg_speeds if s is not None]
    
    if not valid_speeds:
        print("No valid data points to plot.")
        return

    # --- 2. Setup the Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the line (Matplotlib will break the line where data is None)
    line, = ax.plot(frames, avg_speeds, color=line_color, linewidth=2, label='Avg Speed (m/s)')
    
    # Styling
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set axis limits
    ax.set_xlim(min(frames), max(frames))
    y_max = max(valid_speeds)
    ax.set_ylim(0, y_max * 1.1) 

    # --- 3. Initialize Animation Elements ---
    # Vertical cursor line
    cursor = ax.axvline(x=frames[0], color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Text annotation box
    text_annotation = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right')

    # --- 4. Define Update Function ---
    def update(frame_idx):
        current_fid = frames[frame_idx]
        current_speed = avg_speeds[frame_idx]
        
        # Move the red cursor
        cursor.set_xdata([current_fid])
        
        # Update text
        if current_speed is not None:
            status_text = f"Frame: {current_fid}\nSpeed: {current_speed:.4f} m/s"
        else:
            status_text = f"Frame: {current_fid}\nSpeed: No Data"
            
        text_annotation.set_text(status_text)
        
        return cursor, text_annotation

    # --- 5. Create and Save Animation ---
    print(f"Generating GIF with {len(frames)} frames...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames), 
        interval=duration_ms, 
        blit=True
    )
    
    ani.save(output_path, writer='pillow')
    print(f"Saved to {output_path}")
    plt.close()