# Author: Alinjar Dan
# Email: alinjardannitdgp2014@gmail.com
# GitHub: https://github.com/alinjar1996
import numpy as np

def find_points_on_rectangle(rot_axis, x_side, y_side):
    """
    Finds a vector lying in the XY plane (z=0), perpendicular to the given 3D rot_axis,
    passing through the origin and intersecting the rectangle defined by
    x = ±x_side and y = ±y_side.
    """

    r = np.array(rot_axis, dtype=float)
    r /= np.linalg.norm(r)  # normalize rotation axis

    # Construct perpendicular vector in XY plane
    v = np.array([-r[1], r[0], 0.0])

    # Handle degenerate case: if the projection of rot_axis onto XY plane is near zero
    if np.linalg.norm(v) < 1e-8:
        # rotation axis is parallel to z-axis → any vector in XY plane is perpendicular
        v = np.array([1.0, 0.0, 0.0])
    else:
        v /= np.linalg.norm(v)

    # Scale vector so it intersects rectangle boundary
    scale_x = x_side / abs(v[0]) if v[0] != 0 else np.inf
    scale_y = y_side / abs(v[1]) if v[1] != 0 else np.inf
    scale = min(scale_x, scale_y)

    intersection_point = scale * v

    # print("Rotation axis (r):", r)
    # print("XY-plane perpendicular vector (v):", v)
    # print("Intersection point:", intersection_point)

    return v, intersection_point



def main():
    axis = [1.0, 0.0, 1.0]
    axis = axis / np.linalg.norm(axis)
    x_side, y_side = 0.11, 0.15

    v, p = find_points_on_rectangle(axis, x_side, y_side)

    print("v", v)
    print("p", p)



if __name__=="__main__":
    main()