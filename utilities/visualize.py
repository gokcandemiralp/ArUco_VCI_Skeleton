import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
    # 1. Prepare the data
    # Sort frames numerically to ensure the timeline is in order
    sorted_frame_ids = sorted([int(k) for k in differences.keys()])
    
    frames = []
    avg_errors = []
    
    for fid in sorted_frame_ids:
        # Convert back to string key to access the dictionary
        fid_str = str(fid)
        diff_data = differences.get(fid_str, {})
        
        if diff_data:
            avg_error = sum(diff_data.values()) / len(diff_data)
            frames.append(fid)
            avg_errors.append(avg_error)
    
    if not frames:
        print("No data to plot.")
        return

    # 2. Setup the Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the static average error line
    # We plot this once; the animation is just the cursor moving over it.
    ax.plot(frames, avg_errors, color='royalblue', linewidth=2, label='Avg Error (m)')
    
    # Styling
    ax.set_title(f'Average Euclidean Error per Frame', fontsize=14)
    ax.set_xlabel('Frame ID', fontsize=12)
    ax.set_ylabel('Error (meters)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set axis limits with some padding
    ax.set_xlim(min(frames), max(frames))
    y_max = max(avg_errors) if avg_errors else 1.0
    ax.set_ylim(0, y_max * 1.1) 

    # 3. Initialize the moving cursor (vertical line)
    # We initialize it at the first frame
    cursor = ax.axvline(x=frames[0], color='red', linestyle='--', linewidth=2, label='Current Frame')
    
    # Add a text annotation for the current value (optional but helpful)
    text_annotation = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right')

    # 4. Define the Update Function for Animation
    def update(frame_idx):
        # frame_idx is the index in our lists (0 to len(frames)-1)
        current_frame_id = frames[frame_idx]
        current_error = avg_errors[frame_idx]
        
        # Update cursor position
        cursor.set_xdata([current_frame_id, current_frame_id])
        
        # Update text
        text_annotation.set_text(f'Frame: {current_frame_id}\nError: {current_error:.4f} m')
        
        return cursor, text_annotation

    # 5. Generate and Save Animation
    # Calculate FPS based on duration_ms
    fps = 1000.0 / duration_ms
    
    print(f"Generating GIF with {len(frames)} frames at {fps:.2f} FPS...")
    
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(frames), 
        blit=True # Blit optimizes drawing by only re-drawing changed artists
    )
    
    # Use PillowWriter for GIF creation
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    
    plt.close(fig) # Close the plot to free memory
    print(f"Done! GIF saved to: {os.path.abspath(output_path)}")

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