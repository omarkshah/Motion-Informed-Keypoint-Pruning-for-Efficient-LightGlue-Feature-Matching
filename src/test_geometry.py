import numpy as np

from camera import ROS_DEFAULT
from geometry import reproject_keypoints

cam = ROS_DEFAULT

keypoints = np.array([
    [400.0, 300.0],
    [100.0, 200.0],
    [319.5, 239.5]
])

depths = np.array([2.0, 1.0, 3.0])

# identity transformation, should return the same keypoints
result = reproject_keypoints(
    keypoints=keypoints,
    depths=depths,
    src_intrinsics=cam,
    dest_intrinsics=cam,
    R=np.eye(3),
    t=np.zeros(3),
    target_size=(640, 480),
    margin=20
)

np.testing.assert_allclose(
    result.pixels,
    keypoints,
    atol=1e-8
)

assert result.potentially_visible.all()
assert not result.out_of_view.any()

print("Identity test passed")

# camera moves right
result = reproject_keypoints(
    keypoints=np.array([[400.0, 300.0]]),
    depths=np.array([2.0]),
    src_intrinsics=cam,
    dest_intrinsics=cam,
    R=np.eye(3),
    t=np.array([-0.5, 0.0, 0.0]),
    target_size=(640, 480),
    margin=0
)

np.testing.assert_allclose(
    result.pixels,
    [[268.75, 300.0]],
    atol=1e-8
)

assert result.in_front[0]
assert result.in_image[0]

print("Translation test passed")


# camera rotates 180 degrees
R = np.array([
    [-1.0, 0.0,  0.0],
    [ 0.0, 1.0,  0.0],
    [ 0.0, 0.0, -1.0],
])

result = reproject_keypoints(
    keypoints=np.array([[319.5, 239.5]]),
    depths=np.array([2.0]),
    src_intrinsics=cam,
    dest_intrinsics=cam,
    R=R,
    t=np.zeros(3),
    target_size=(640, 480)
)

np.testing.assert_allclose(
    result.target_depths,
    [-2.0]
)

assert result.behind[0]
assert not result.in_front[0]
assert result.out_of_view[0]
assert np.isnan(result.pixels).all()

print("Behind-camera test passed")

# missing depth
result = reproject_keypoints(
    keypoints=np.array([[400.0, 300.0]]),
    depths=np.array([np.nan]),
    src_intrinsics=cam,
    dest_intrinsics=cam,
    R=np.eye(3),
    t=np.zeros(3),
    target_size=(640, 480)
)

assert not result.has_depth[0]
assert result.unknown[0]
assert not result.out_of_view[0]
assert result.keep_mask[0]

print("Missing-depth test passed")

# safety margin
result = reproject_keypoints(
    keypoints=np.array([[100.0, 300.0]]),
    depths=np.array([2.0]),
    src_intrinsics=cam,
    dest_intrinsics=cam,
    R=np.eye(3),
    t=np.array([-0.4, 0.0, 0.0]),
    target_size=(640, 480),
    margin=20
)

np.testing.assert_allclose(
    result.pixels,
    [[-5.0, 300.0]],
    atol=1e-8
)

assert not result.in_image[0]
assert result.within_margin[0]
assert result.keep_mask[0]

print("Safety-margin test passed")