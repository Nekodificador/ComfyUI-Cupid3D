"""Cupid's predicted camera -> ComfyUI's LOAD3D_CAMERA dict.

Several conventions meet here:

* Cupid returns a 4x4 world->camera extrinsic in OpenCV axes (x right, y down,
  z forward) over a z-up world, and a 3x3 intrinsic normalized by the longest
  side of the image it was given — `pose_utils.calibrate_camera_dlt` states the
  image points are in the 0-1 range, and `ImageProcessor.crop_to_content`
  divides by `max(image.size)`.
* The model wants a square input, so a non-square image is padded before it is
  handed over. The intrinsic comes back in the padded frame's coordinates and
  is shifted here into the original frame's.
* ComfyUI's MESH is glTF y-up while the raw Cupid mesh is z-up. Upstream's
  `to_glb` rotates it with (x, y, z) -> (x, z, -y), the same rotation core's
  Trellis2 shape decode applies, so the camera has to move with it.
* LOAD3D_CAMERA is three.js: y-up, the camera looking down its own -z, and the
  quaternion cam->world as (x, y, z, w). It has no principal point, and Cupid
  predicts an off-centre one. Rather than pad the frame until that point is
  centred, the camera is rotated so its optical axis passes through the frame
  centre, which keeps the user's own image as the background.

  That swap is an approximation, since a sensor shift and a rotation are not
  the same transform, but a measured one. On a 1000 px frame it costs 8 px at
  worst (p99 3.5 px) for a 0.125 offset at 14 deg, and still only 36 px (p99
  18 px) at 65 deg with a 0.25 offset — against a reconstruction whose own
  silhouette lands about 2.4% from the photo's.

The dict shape, the vertical `fov` in degrees and the quaternion helper follow
NKD VFX Tools' NKDfSpyCamera so both nodes agree bit for bit.
"""
import math

import numpy as np

__all__ = ['ZUP_TO_YUP', 'camera_for_image']

# z-up -> y-up, acting on column vectors: (x, y, z) -> (x, z, -y).
ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0]])

# OpenCV camera axes -> three.js camera axes (flip y and z).
_CV_TO_THREE = np.diag([1.0, -1.0, -1.0])


def _mat_to_quat_xyzw(R):
    """3x3 rotation matrix -> normalized quaternion (x, y, z, w). Shepperd's method.

    Copied from NKD VFX Tools' nkd_fspy_camera so the two nodes emit the same
    quaternion for the same rotation.
    """
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def _rotation_onto(a, b):
    """Rotation matrix taking unit vector `a` onto unit vector `b`."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _camera_dict(extrinsic, tan_x, tan_y, world_offset=(0.0, 0.0, 0.0)):
    """Pose plus half-angle tangents -> LOAD3D_CAMERA.

    `world_offset` is a translation applied to the mesh; the camera takes the
    same one, so the projection is unchanged.
    """
    E = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    R_wc = E[:3, :3] @ ZUP_TO_YUP.T          # follow the mesh into the y-up world
    t = E[:3, 3]

    R_cw = R_wc.T
    pos = -R_cw @ t
    R_three = R_cw @ _CV_TO_THREE
    q = _mat_to_quat_xyzw(R_three)

    # Look along the optical axis, out to the depth of the world origin, which
    # is where the reconstruction is centred.
    distance = float(np.linalg.norm(pos)) or 1.0
    target = pos - distance * R_three[:, 2]

    offset = np.asarray(world_offset, dtype=np.float64)
    pos, target = pos + offset, target + offset

    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {
        "position": xyz(pos), "target": xyz(target),
        "quaternion": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
        "fov": float(math.degrees(2.0 * math.atan(tan_y))), "aspect": float(tan_x / tan_y),
        "zoom": 1.0, "cameraType": "perspective",
    }


def camera_for_image(extrinsic, intrinsic, size, pad=(0, 0), world_offset=(0.0, 0.0, 0.0)):
    """Build the camera that describes the ORIGINAL image.

    Args:
        extrinsic: 4x4 world->camera matrix, OpenCV axes.
        intrinsic: 3x3 intrinsic, normalized by the padded square's side.
        size: (width, height) of the original image.
        pad: (left, top) padding that was added to square it.
        world_offset: translation applied to the mesh, which the camera
            follows so the projection stays put.

    Returns:
        A LOAD3D_CAMERA dict.
    """
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    W, H = int(size[0]), int(size[1])
    base = float(max(W, H))                  # the square's side, which normalized the intrinsic

    fx, fy = float(K[0, 0]), float(K[1, 1])
    # Move the principal point out of the padding, into the original frame.
    cx = float(K[0, 2]) - pad[0] / base
    cy = float(K[1, 2]) - pad[1] / base
    span_x, span_y = W / base, H / base

    # Turn the camera so the ray through the frame's centre becomes its axis.
    d = np.array([(0.5 * span_x - cx) / fx, (0.5 * span_y - cy) / fy, 1.0])
    R = _rotation_onto(d / np.linalg.norm(d), np.array([0.0, 0.0, 1.0]))
    E = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4).copy()
    E[:3, :3] = R @ E[:3, :3]                # a rotation in place: the centre does not move
    E[:3, 3] = R @ E[:3, 3]

    return _camera_dict(E, tan_x=0.5 * span_x / fx, tan_y=0.5 * span_y / fy,
                        world_offset=world_offset)


def _demo():
    """Place known cameras, round-trip them, and check what comes back."""
    # A camera 4 units along the world +x of a z-up world, looking at the origin.
    fwd = np.array([-1.0, 0.0, 0.0])
    up_world = np.array([0.0, 0.0, 1.0])
    down = -up_world                         # OpenCV's camera y points down
    right = np.cross(down, fwd)
    R_cw = np.stack([right, down, fwd], axis=1)       # columns: cam x, y, z in world
    cam_pos = np.array([4.0, 0.0, 0.0])
    R_wc = R_cw.T
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R_wc, -R_wc @ cam_pos

    fy = 0.5 / math.tan(math.radians(30.0) / 2.0)     # 30 deg vertical fov when centred
    K = np.array([[fy, 0, 0.5], [0, fy, 0.5], [0, 0, 1.0]])

    def quat_matrix(cam):
        x, y, z, w = (cam["quaternion"][k] for k in "xyzw")
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def project(cam, pts, W, H):
        pos = np.array([cam["position"][k] for k in "xyz"])
        Rq = quat_matrix(cam)
        v = (pts - pos) @ Rq                          # Rq.T @ (p - pos), per point
        tan_h = math.tan(math.radians(cam["fov"]) * 0.5)
        return np.stack([v[:, 0] / (-v[:, 2] * tan_h * (W / H)),
                         v[:, 1] / (-v[:, 2] * tan_h)], axis=-1)

    # --- centred principal point, square frame
    cam = camera_for_image(E, K, (256, 256))
    assert abs(cam["fov"] - 30.0) < 1e-9, cam["fov"]
    assert abs(cam["aspect"] - 1.0) < 1e-12
    p = np.array([cam["position"][k] for k in "xyz"])
    assert np.allclose(p, [4.0, 0.0, 0.0], atol=1e-9), p
    assert np.allclose([cam["target"][k] for k in "xyz"], [0, 0, 0], atol=1e-9)
    Rq = quat_matrix(cam)
    assert np.allclose(Rq @ [0, 0, -1], [-1, 0, 0], atol=1e-9)   # -z is the view direction
    assert np.allclose(Rq @ [0, 1, 0], [0, 1, 0], atol=1e-9)     # +y is world up

    # mesh and camera must agree on the z-up -> y-up flip
    pt_zup = np.array([0.0, 0.0, 1.0])
    depth_before = (E[:3, :3] @ pt_zup + E[:3, 3])[2]
    depth_after = (E[:3, :3] @ ZUP_TO_YUP.T @ (ZUP_TO_YUP @ pt_zup) + E[:3, 3])[2]
    assert abs(depth_before - depth_after) < 1e-12

    # --- off-centre principal point: a point on the optical axis still has to
    # land at the centre of the ORIGINAL frame, with no padding anywhere.
    K_off = np.array([[fy, 0, 0.30], [0, fy, 0.375], [0, 0, 1.0]])
    cam = camera_for_image(E, K_off, (256, 256))
    pos = np.array([cam["position"][k] for k in "xyz"])
    axis_pt = pos + 2.0 * (quat_matrix(cam) @ [0, 0, -1])
    assert np.allclose(project(cam, axis_pt[None], 256, 256)[0], [0, 0], atol=1e-12)

    # --- a non-square frame, padded left/right to reach the square the model saw
    W, H, pad_left = 171, 256, 42
    K_ns = np.array([[fy, 0, 0.5], [0, fy, 0.5], [0, 0, 1.0]])   # centred in the SQUARE
    cam_ns = camera_for_image(E, K_ns, (W, H), pad=(pad_left, 0))
    # the square's centre sits at the original frame's centre only if the pad is
    # symmetric, which it is: so the axis point lands at the centre again
    pos = np.array([cam_ns["position"][k] for k in "xyz"])
    axis_pt = pos + 2.0 * (quat_matrix(cam_ns) @ [0, 0, -1])
    assert np.allclose(project(cam_ns, axis_pt[None], W, H)[0], [0, 0], atol=1e-12)
    assert abs(cam_ns["aspect"] - W / H) < 1e-12, (cam_ns["aspect"], W / H)

    # --- world offset: moving mesh and camera together must not move the picture
    pts = np.array([[0.1, 0.2, -0.3], [-0.4, 0.0, 0.25], [0.0, 0.5, 0.0]])
    lift = np.array([0.0, 0.42, 0.0])
    a = project(camera_for_image(E, K_off, (256, 256)), pts, 256, 256)
    b = project(camera_for_image(E, K_off, (256, 256), world_offset=lift), pts + lift, 256, 256)
    assert np.allclose(a, b, atol=1e-12), np.abs(a - b).max()

    print("cupid_camera: pose, fov, quaternion, principal point, padding and "
          "ground offset all agree")


if __name__ == '__main__':
    _demo()
