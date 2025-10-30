# Author: Iryna hurova
# Email: iryna.gurova@gmail.com
# GitHub: https://github.com/patsyuk03/
import numpy as np
import jax.numpy as jnp

def quat_to_rotmat(quat):
    """
    Convert a quaternion (w, x, y, z) to a 1x9 rotation matrix.
    Assumes the quaternion is normalized.
    """
    w, x, y, z = quat

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z

    yz = y * z

    rot_mat = np.array([
        [ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz]
    ]).flatten()

    return rot_mat

def rotmat_to_quat(mat):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).
    Assumes the matrix is a valid rotation matrix.
    """
    m = mat.reshape((3, 3))
    tr = m[0, 0] + m[1, 1] + m[2, 2]

    def case_tr_pos(_):
        S = jnp.sqrt(tr + 1.0) * 2  # S=4*w
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
        return jnp.array([w, x, y, z])

    def case_m00_max(_):
        S = jnp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # S=4*x
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
        return jnp.array([w, x, y, z])

    def case_m11_max(_):
        S = jnp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # S=4*y
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
        return jnp.array([w, x, y, z])

    def case_m22_max(_):
        S = jnp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # S=4*z
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
        return jnp.array([w, x, y, z])

    quat = jnp.where(tr > 0, case_tr_pos(0),
            jnp.where((m[0, 0] > m[1, 1]) & (m[0, 0] > m[2, 2]), case_m00_max(0),
            jnp.where(m[1, 1] > m[2, 2], case_m11_max(0), case_m22_max(0))
        ))

    return quat

def quaternion_distance(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)

def rotation_quaternion(angle_deg, axis):
    """ 
    Creates rotation quaternion. 

    Args: 
    angle_deg: Degrees to turn. 
    axis: Along which axis to turn (one at a time) [x, y, z]. 

    Returns: 
    Rotation quaternion [w, x, y, z]. 
    """ 

    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    w = np.cos(angle_rad / 2)
    x, y, z = axis * np.sin(angle_rad / 2)
    return np.array([round(w, 5), round(x, 5), round(y, 5), round(z, 5)])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    
    return np.array([round(w, 5), round(x, 5), round(y, 5), round(z, 5)])

def angle_between_lines(p1, p2, p3, p4): 
    """ 
    Calculates the angle between two lines using NumPy. 

    Args: 
    p1, p2: Endpoints of the first line ((x1, y1), (x2, y2)). 
    p3, p4: Endpoints of the second line ((x3, y3), (x4, y4)). 

    Returns: 
    The angle in degrees between the two lines. 
    """ 
    # Create vectors from the points 
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]]) 
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]]) 

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    epsilon = 0.1


    if norm_v1 < epsilon or norm_v2 < epsilon:
        # print("EXCEPTION 1", flush=True)
        return 0
    
    u1 = v1 / norm_v1
    u2 = v2 / norm_v2

    dot_product = np.dot(u1, u2)

    if np.abs(dot_product-1)<0.00001:
        # print("EXCEPTION 2", flush=True)
        return 0

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # print(f"ANGLES: {angle1:.2f}, {angle2:.2f} | Vs: {v1, v2}| Norm: {norm_v1:.2f}, {norm_v2:.2f} | d: {dot_product:.2f}", flush=True)
    
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad)

    return angle_deg



def turn_quat(eef_pos_1_init, eef_pos_2_init, eef_pos_1, eef_pos_2, tray_rot_init):
    # # xy plane z-axis rotation
    # p1, p2 = (eef_pos_1_init[0], eef_pos_1_init[1]), (eef_pos_2_init[0], eef_pos_2_init[1])
    # p3, p4 = (eef_pos_1[0], eef_pos_1[1]), (eef_pos_2[0], eef_pos_2[1])
    # z_rot = angle_between_lines(p1, p2, p3, p4)
    # tray_rot = quaternion_multiply(tray_rot_init, rotation_quaternion(z_rot, [0, 0, 1]))

    # xz plane y-axis rotation
    p1, p2 = (eef_pos_1_init[2], eef_pos_1_init[0]), (eef_pos_2_init[2], eef_pos_2_init[0])
    p3, p4 = (eef_pos_1[2], eef_pos_1[0]), (eef_pos_2[2], eef_pos_2[0])
    y_rot = angle_between_lines(p1, p2, p3, p4)
    tray_rot = quaternion_multiply(tray_rot, rotation_quaternion(y_rot, [0, 1, 0]))

    # # yz plane x-axis rotation
    # p1, p2 = (eef_pos_1_init[1], eef_pos_1_init[2]), (eef_pos_2_init[1], eef_pos_2_init[2])
    # p3, p4 = (eef_pos_1[1], eef_pos_1[2]), (eef_pos_2[1], eef_pos_2[2])
    # x_rot = angle_between_lines(p1, p2, p3, p4)
    # tray_rot = quaternion_multiply(tray_rot, rotation_quaternion(x_rot, [1, 0, 0]))

    # np.set_printoptions(precision=2, suppress=True)
    # print(f"XY-coords:  (({p1[0]:.2f}, {p1[1]:.2f}), ({p2[0]:.2f}, {p2[1]:.2f})), (({p3[0]:.2f}, {p3[1]:.2f}), ({p4[0]:.2f}, {p4[1]:.2f})) | XYZ-Axis rotation: {float(y_rot), float(z_rot)}", flush=True)

    return tray_rot



def turn_quat_general(p1_init, p2_init, p1_target, p2_target, tray_rot_init):
    # 1. Define the Initial and Target Vectors
    # The vectors represent the line segment connecting the two points
    
    # Vector 1 (Initial Orientation)
    v_init = p2_init - p1_init
    # Vector 2 (Target Orientation)
    v_target = p2_target - p1_target

    # Normalize the vectors to ensure accurate calculations
    v_init_norm = v_init / np.linalg.norm(v_init)
    v_target_norm = v_target / np.linalg.norm(v_target)

    # 2. Calculate the Axis of Rotation (Axis-Angle Representation)
    
    # The axis of rotation is perpendicular to both the initial and target vectors.
    # This is found using the Cross Product.
    rotation_axis = np.cross(v_init_norm, v_target_norm)

    # 3. Calculate the Angle of Rotation
    
    # The cosine of the angle between the vectors is found using the Dot Product.
    cos_angle = np.dot(v_init_norm, v_target_norm)
    
    # Clamp the value for numerical stability (in case cos_angle is slightly > 1 or < -1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0) 
    
    rotation_angle = np.arccos(cos_angle) # The angle in radians

    # 4. Handle Edge Cases (Parallel or Anti-Parallel Vectors)
    
    # If the vectors are already aligned (angle is 0 or 180 degrees), the cross product is zero.
    # We use a small epsilon to check if the axis is zero (i.e., vectors are parallel/anti-parallel).
    if np.linalg.norm(rotation_axis) < 1e-6:
        if rotation_angle < 1e-6: # Already aligned (0 degrees)
            # No rotation needed, return initial quaternion
            return tray_rot_init 
        else: # Anti-parallel (180 degrees)
            # The rotation is 180 degrees. The axis can be arbitrary (e.g., perpendicular to v_init).
            # We can use the shortest axis that is not collinear with v_init.
            
            # Find a non-collinear vector (e.g., check x, y, z components)
            temp_vec = np.array([1, 0, 0])
            if np.abs(np.dot(temp_vec, v_init_norm)) > 0.9: # If v_init is close to X-axis
                temp_vec = np.array([0, 1, 0])
            
            # The new axis is perpendicular to v_init and temp_vec
            rotation_axis = np.cross(v_init_norm, temp_vec)


    # Normalize the final rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    print("rotation_angle", rotation_angle)
    print("rotation_axis", rotation_axis)

    rotation_angle_deg = np.rad2deg(rotation_angle)

    # 5. Create and Apply the Rotation Quaternion
    
    # Convert the calculated axis and angle into a rotation quaternion
    q_change = rotation_quaternion(rotation_angle_deg, rotation_axis)
    
    # Combine the initial rotation with the new change
    tray_rot_new = quaternion_multiply(tray_rot_init, q_change)

    return tray_rot_new


def main():

    # quat0 = np.array([1, 0, 0, 0])
    # quat1 = quaternion_multiply(quat0, rotation_quaternion(-180, [0, 1, 0])) # rotate wuaternion -180 degrees along y-axis
    # quat2 = quaternion_multiply(quat1, rotation_quaternion(-90, [0, 0, 1])) # rotate wuaternion -90 degrees along z-axis
    # mat0 = quat_to_rotmat(quat2) # convert quaternion to rotation matrix
    # quat3 = rotmat_to_quat(mat0) # convert rotation matrix back to quaternion

    # print(quat0)
    # print(quat1)
    # print(quat2)
    # print(mat0)
    # print(quat3)

    # Initial state (Identity quaternion: [1, 0, 0, 0] means 0 rotation)
    TRAY_ROT_INIT = np.array([1.0, 0.0, 0.0, 0.0]) 

    print("--- Running turn_quat_general Tests ---")
    
    # ======================================================================
    # CASE 1: 90-Degree Rotation (Around Z-axis)
    # The tray rotates 90 degrees around Z to align X-vector with Y-vector.
    # ======================================================================
    P1_INIT_1 = np.array([0., 0., 0.])
    P2_INIT_1 = np.array([0., 1., 0.])  
    P1_TARGET_1 = np.array([0., 0., 0.])
    P2_TARGET_1 = np.array([0., 0., 1.])  
    
    q_1 = turn_quat_general(P1_INIT_1, P2_INIT_1, P1_TARGET_1, P2_TARGET_1, TRAY_ROT_INIT)
    # euler_1 = quat_to_euler(q_1) # [Roll, Pitch, Yaw]
    print("q1", q_1)


if __name__=="__main__":
    main()