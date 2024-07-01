import numpy as np


class ModelDoubleSphereNumpy:
    """Implemented according to:
    V. Usenko, N. Demmel, and D. Cremers: The Double Sphere Camera Model.
    Proceedings of the International Conference on 3D Vision (3DV) (2018).
    URL: https://arxiv.org/pdf/1807.08957.pdf."""

    def __init__(self, xi, alpha, fx, fy, cx, cy, shape_image):
        self.alpha = alpha
        self.cx = cx
        self.cy = cy
        self.device = None
        self.fx = fx
        self.fy = fy
        self.shape_image = shape_image
        self.xi = xi

    @classmethod
    def from_camera_info_message(cls, message):
        try:
            binning_x = message.binning_x if message.binning_x != 0 else 1
            binning_y = message.binning_y if message.binning_y != 0 else 1
            offset_x = message.roi.offset_x
            offset_y = message.roi.offset_y
        except AttributeError:
            binning_x = 1
            binning_y = 1
            offset_x = 0
            offset_y = 0

        # Do not know channel dimension from camera info message
        shape_image = (-1, message.height, message.width)

        xi = message.d[0]
        alpha = message.d[1]
        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        instance = cls(xi, alpha, fx, fy, cx, cy, shape_image)
        return instance

    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_xyz = coords_xyz.astype(np.float16)

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        # Eq. (41)
        d1 = np.sqrt(x**2 + y**2 + z**2)
        # Eq. (45)
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        # Eq. (44)
        w2 = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi**2 + 1.0)
        # Eq. (43)
        mask_valid = z > -w2 * d1

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            x = x[mask_valid][None, ...]
            y = y[mask_valid][None, ...]
            z = z[mask_valid][None, ...]
            d1 = d1[mask_valid][None, ...]
            mask_valid = np.ones_like(z, dtype=bool)

        # Eq. (42)
        z_shifted = self.xi * d1 + z
        d2 = np.sqrt(x**2 + y**2 + z_shifted**2)
        # Eq. (40)
        denominator = self.alpha * d2 + (1 - self.alpha) * z_shifted
        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy
        coords_uv = np.stack([u, v], dim=1)

        if use_mask_fov:
            mask_left = coords_uv[:, 0, :] >= 0
            mask_top = coords_uv[:, 1, :] >= 0
            mask_right = coords_uv[:, 0, :] < self.shape_image[2]
            mask_bottom = coords_uv[:, 1, :] < self.shape_image[1]
            mask_valid *= mask_left * mask_top * mask_right * mask_bottom

        return coords_uv, mask_valid

    def project_image_onto_points(self, coords_uv, use_invalid_coords=True, use_half_precision=True):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_xyz = coords_xyz.as_type(np.float16)

        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        # Eq. (47)
        mx = (u - self.cx) / self.fx
        # Eq. (48)
        my = (v - self.cy) / self.fy
        # Eq. (49)
        square_r = mx**2 + my**2
        # Eq. (51) can be written to use this
        term = 1.0 - (2.0 * self.alpha - 1.0) * square_r
        # Eq. (51)
        mask_valid = term >= 0 if self.alpha > 0.5 else np.ones_like(term, dtype=bool)

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            mx = mx[mask_valid][None, ...]
            my = my[mask_valid][None, ...]
            square_r = square_r[mask_valid][None, ...]
            term = term[mask_valid][None, ...]
            mask_valid = np.ones_like(term, dtype=bool)

        # Eq. (50)
        mz = (1.0 - self.alpha**2 * square_r) / (self.alpha * np.sqrt(term) + 1.0 - self.alpha)
        # Eq. (46)
        factor = (mz * self.xi + np.sqrt(mz**2 + (1.0 - self.xi**2) * square_r)) / (mz**2 + square_r)
        coords_xyz = factor[:, None, :] * np.stack((mx, my, mz), dim=1)
        coords_xyz[:, 2, :] -= self.xi

        return coords_xyz, mask_valid
