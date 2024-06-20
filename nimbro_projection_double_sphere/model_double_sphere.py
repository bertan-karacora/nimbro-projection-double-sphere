import numpy as np
import torch


class ModelDoubleSphere:
    """Implemented according to:
    V. Usenko, N. Demmel, and D. Cremers: The Double Sphere Camera Model.
    Proceedings of the International Conference on 3D Vision (3DV) (2018).
    URL: https://arxiv.org/pdf/1807.08957.pdf."""

    def __init__(self, alpha, xi, fx, fy, cx, cy, device=None):
        self.alpha = 0.62011029
        self.cx = 1335.9318658
        self.cy = 984.09536152
        self.fx = 766.54936879
        self.fy = 766.48181735
        self.xi = -0.04469921
        # -0.04469921 0.62011029  766.54936879  766.48181735 1335.9318658 984.09536152

    @classmethod
    def from_camera_info_message(cls, message, device=None):
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

        xi = message.d[0]
        alpha = message.d[1]
        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        instance = cls(xi, alpha, fx, fy, cx, cy, device=device)
        return instance

    @torch.inference_mode()
    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True):
        """Project 3D points onto 2D image.
        Points frame: [right, down, front].
        Image frame: [right, down]."""
        x, y, z = coords_xyz[..., 0, :], coords_xyz[..., 1, :], coords_xyz[..., 2, :]

        # Eq. (41)
        d1 = torch.sqrt(x**2 + y**2 + z**2)
        # Eq. (45)
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        # Eq. (44)
        w2 = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi**2 + 1.0)

        # Eq. (43)
        mask_valid = z > -w2 * d1
        if not use_invalid_coords:
            points = points.permute(0, 2, 1)
            points = points[mask_valid]
            points = points.permute(0, 2, 1)
            x = x[mask_valid]
            y = y[mask_valid]
            z = z[mask_valid]
            d1 = d1[mask_valid]
            mask_valid = torch.ones_like(z, dtype=torch.bool)

        # Eq. (42)
        z_shifted = self.xi * d1 + z
        d2 = torch.sqrt(x**2 + y**2 + z_shifted**2)

        # Eq. (40)
        denominator = self.alpha * d2 + (1 - self.alpha) * z_shifted
        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy
        coords_uv = torch.stack([u, v], dim=-1)

        return coords_uv, mask_valid

    @torch.inference_mode()
    def project_image_onto_points(self, coords_uv, use_invalid_coords=True):
        """Project 2D image onto 3D unit sphere.
        Points coordinate system: [right, down, front]
        Image coordinate system: [right, down]"""
        u, v = coords_uv[..., 0, :], coords_uv[..., 1, :]

        # Eq. (47)
        mx = (u - self.cx) / self.fx
        # Eq. (48)
        my = (v - self.cy) / self.fy
        # Eq. (49)
        square_r = mx**2 + my**2

        # Eq. (51) can be written to use this
        term = 1.0 - (2.0 * self.alpha - 1.0) * square_r

        # Eq. (51)
        mask_valid = term >= 0 if self.alpha > 0.5 else torch.ones_like(term, dtype=torch.bool)
        if not use_invalid_coords:
            mx = mx[mask_valid]
            my = my[mask_valid]
            square_r = square_r[mask_valid]
            term = term[mask_valid]
            mask_valid = torch.ones_like(term, dtype=torch.bool)

        # Eq. (50)
        mz = (1.0 - self.alpha**2 * square_r**2) / (self.alpha * np.sqrt(term) + 1.0 - self.alpha)

        # Eq. (46)
        factor = (mz * self.xi + np.sqrt(mz**2 + (1.0 - self.xi**2) * square_r)) / (mz**2 + square_r)
        coords_xyz = factor * torch.stack([mx, my, mz], dim=-1)
        coords_xyz[..., -1] -= self.xi

        return coords_xyz, mask_valid

    # def _warp_img(self, img, img_pts, valid_mask):
    #     # Remap
    #     img_pts = img_pts.astype(np.float32)
    #     out = cv2.remap(img, img_pts[..., 0], img_pts[..., 1], cv2.INTER_LINEAR)
    #     out[~valid_mask] = 0.0
    #     return out

    # def to_perspective(self, img, img_size=(512, 512), f=0.25):
    #     # Generate 3D points
    #     h, w = img_size
    #     z = f * min(img_size)
    #     x = np.arange(w) - w / 2
    #     y = np.arange(h) - h / 2
    #     x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    #     point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)], axis=-1)

    #     # Project on image plane
    #     img_pts, valid_mask = self.world2cam(point3D)
    #     out = self._warp_img(img, img_pts, valid_mask)
    #     return out

    # def to_equirect(self, img, img_size=(256, 512)):
    #     # Generate 3D points
    #     h, w = img_size
    #     phi = -np.pi + (np.arange(w) + 0.5) * 2 * np.pi / w
    #     theta = -np.pi / 2 + (np.arange(h) + 0.5) * np.pi / h
    #     phi_xy, theta_xy = np.meshgrid(phi, theta, indexing="xy")

    #     x = np.sin(phi_xy) * np.cos(theta_xy)
    #     y = np.sin(theta_xy)
    #     z = np.cos(phi_xy) * np.cos(theta_xy)
    #     point3D = np.stack([x, y, z], axis=-1)

    #     # Project on image plane
    #     img_pts, valid_mask = self.world2cam(point3D)
    #     out = self._warp_img(img, img_pts, valid_mask)
    #     return out
