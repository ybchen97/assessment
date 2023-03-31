import math
import open3d as o3d
import numpy as np
from spatialmath import SE3, UnitQuaternion
import spatialmath.base as smb

"""
Our camera wireframe has a total of 5 points:
- point 0 for the camera center
- points 1 to 4 for the viewing port

Suppose that the camera is pointing in the positive x-axis direction. Then,
define the viewing port to have height of 1 unit and width of 2 units, and a
perpendicular distance of 0.5 unit from camera center. The other points can be
defined relative to the camera center point 0:
- point 1: top-left,  translated by (x,y,z) = (0.25,  0.25, -0.5)
- point 2: bot-left,  translated by (x,y,z) = (0.25, -0.25, -0.5)
- point 3: bot-right, translated by (x,y,z) = (0.25, -0.25,  0.5)
- point 4: top-right, translated by (x,y,z) = (0.25,  0.25,  0.5)

Camera Trajectory:
Moving the camera from the origin onto point (x,y,z) = (2,2,3). This can be
achieved by simply translating the camera from the origin to (x,y,z) = (2,2,3),
while also ensuring that the camera turns to face the object.
"""

# Relative configurations of points from camera center
PT_1_FROM_C = SE3(0.25,  0.25, -0.5)  # top-left
PT_2_FROM_C = SE3(0.25, -0.25, -0.5)  # bot-left
PT_3_FROM_C = SE3(0.25, -0.25,  0.5)  # bot-right
PT_4_FROM_C = SE3(0.25,  0.25,  0.5)  # top-right

# Sphere object pose
SPHERE_POSE = SE3(2,2,2)
INITIAL_CAMERA_POSE = SE3()

def compute_wireframe_points(pose):
    """Given camera pose, compute wireframe points based on relative
    configurations of viewing port points from camera center."""
    pt_1 = (pose * PT_1_FROM_C).t
    pt_2 = (pose * PT_2_FROM_C).t
    pt_3 = (pose * PT_3_FROM_C).t
    pt_4 = (pose * PT_4_FROM_C).t
    center = pose.t

    return np.array([center, pt_1, pt_2, pt_3, pt_4])

def create_wireframe_object(pose):
    wireframe_points = compute_wireframe_points(pose)

    # Our lines span from point 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(wireframe_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_camera_trajectory(num_steps):
    """
    """

    rotations = []
    translations = []

    # Camera is "facing" the x-axis. Rotate 90 degress counter-clockwise in
    # y-axis to face the sphere
    rot_step_1 = UnitQuaternion.Ry(90 / num_steps, 'deg')
    rotations += [rot_step_1.A] * num_steps

    return np.array(rotations), np.array(translations)

def main():
    # Create coordinate frame object
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,
                                                              origin=[0,0,0])

    # Create sphere object for camera to rotate around
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.1, 0.1, 0.7])
    
    # Since sphere is created at origin, move it to (2,2,2) in world frame
    sphere.transform(SPHERE_POSE.A)

    # Create a camera wireframe object, which is a LineSet object
    wireframe = create_wireframe_object(INITIAL_CAMERA_POSE)

    # Initialize Visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(frame)
    vis.add_geometry(sphere)
    vis.add_geometry(wireframe)

    camera_center = np.array([0.0,0.0,0.0])
    curr_camera_direction = np.array([1.0,0.0,0.0])
    num_steps = 1000

    for i in range(num_steps):
        # Moving camera center to (x,y,z) = (2,2,3)
        # Since camera center is initially at origin, translation vector is just
        # (2,2,3). Then, each step marches an equal distance in the direction of
        # (2,2,3).
        translation_step = np.array([2,2,3]) / num_steps

        # update camera center
        camera_center += translation_step

        # Translate camera
        wireframe.translate(translation_step)
        
        # Determine rotation between two vectors
        target_camera_direction = SPHERE_POSE.t - camera_center
        target_camera_direction /= smb.norm(target_camera_direction)
        curr_camera_direction /= smb.norm(curr_camera_direction)

        axis = smb.cross(curr_camera_direction, target_camera_direction)

        dot_product = np.dot(target_camera_direction, curr_camera_direction)
        angle = math.acos(dot_product)

        rotation_step = smb.angvec2r(angle, axis)
        
        # Update current camera direction
        curr_camera_direction = np.matmul(rotation_step, curr_camera_direction)

        # Rotate camera about camera center
        wireframe.rotate(rotation_step, camera_center)

        # Update visualizer
        vis.update_geometry(wireframe)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    main()
