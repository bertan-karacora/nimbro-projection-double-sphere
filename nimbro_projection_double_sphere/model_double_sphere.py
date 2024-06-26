import numpy as np
import torch
import torch_geometric.nn.unpool as geometric_unpool


class ModelDoubleSphere:
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        shape_image = (message.height, message.width)

        xi = message.d[0]
        alpha = message.d[1]
        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        instance = cls(xi, alpha, fx, fy, cx, cy, shape_image)
        return instance

    @torch.inference_mode()
    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True, use_mask_fov=True):
        """Project 3D points onto 2D image.
        Points frame: [right, down, front].
        Image frame: [right, down]."""
        coords_xyz = coords_xyz.half().to(self.device)

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        # Eq. (41)
        d1 = torch.sqrt(x**2 + y**2 + z**2)
        # Eq. (45)
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        # Eq. (44)
        w2 = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi**2 + 1.0)

        # Eq. (43)
        mask_valid = z > -w2 * d1

        # Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
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
        coords_uv = torch.stack([u, v], dim=1)

        if use_mask_fov:
            mask_left = coords_uv[:, 0, :] >= 0
            mask_top = coords_uv[:, 1, :] >= 0
            mask_right = coords_uv[:, 0, :] < self.shape_image[1]
            mask_bottom = coords_uv[:, 1, :] < self.shape_image[0]
            mask_valid *= mask_left * mask_top * mask_right * mask_bottom

        return coords_uv, mask_valid

    @torch.inference_mode()
    def project_image_onto_points(self, coords_uv, use_invalid_coords=True):
        """Project 2D image onto 3D unit sphere.
        Points coordinate system: [right, down, front]
        Image coordinate system: [right, down]"""
        coords_uv = coords_uv.half().to(self.device)

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
        mask_valid = term >= 0 if self.alpha > 0.5 else torch.ones_like(term, dtype=torch.bool)

        # Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            mx = mx[mask_valid]
            my = my[mask_valid]
            square_r = square_r[mask_valid]
            term = term[mask_valid]
            mask_valid = torch.ones_like(term, dtype=torch.bool)

        # Eq. (50)
        mz = (1.0 - self.alpha**2 * square_r) / (self.alpha * torch.sqrt(term) + 1.0 - self.alpha)

        # Eq. (46)
        factor = (mz * self.xi + torch.sqrt(mz**2 + (1.0 - self.xi**2) * square_r)) / (mz**2 + square_r)
        coords_xyz = factor[:, None, :] * torch.stack([mx, my, mz], dim=1)
        coords_xyz[:, 2, :] -= self.xi

        return coords_xyz, mask_valid

    @torch.inference_mode()
    def sample_color(self, coords_uv, image, mask_valid, color_invalid=(255, 87, 51)):
        coords_uv = coords_uv.half().to(self.device)
        image = image.half().to(self.device)
        mask_valid = mask_valid.to(self.device)

        coords_image = coords_uv.clone()
        coords_image[:, 0, :] = 2.0 * coords_image[:, 0, :] / self.shape_image[1] - 1.0
        coords_image[:, 1, :] = 2.0 * coords_image[:, 1, :] / self.shape_image[0] - 1.0

        coords_image = coords_image.permute(0, 2, 1)
        # grid_sample not implemented for dtype byte
        colors = torch.nn.functional.grid_sample(input=image, grid=coords_image[:, None, :, :], align_corners=True)
        colors = colors[:, :, 0, :]

        colors[:, 0, :].masked_fill_(~mask_valid, color_invalid[0])
        colors[:, 1, :].masked_fill_(~mask_valid, color_invalid[1])
        colors[:, 2, :].masked_fill_(~mask_valid, color_invalid[2])

        return colors

    # Only working for batchsize 1.
    @torch.inference_mode()
    def sample_depth(self, coords_uv, points, mask_valid, use_knn_interpolate=True, ratio_downsampling=8):
        coords_uv = coords_uv.to(self.device)
        points = points.to(self.device)
        mask_valid = mask_valid.to(self.device)

        coords_uv = coords_uv.permute(0, 2, 1)
        coords_uv = coords_uv[mask_valid]
        coords_uv = coords_uv.permute(1, 0)

        # Note: values get rounded
        coords_uv = coords_uv.long()
        u, v = coords_uv[0], coords_uv[1]
        coords_uvflat = v * self.shape_image[1] + u
        coords_uvflat = coords_uvflat.view(-1)

        z = points[:, 2, :]
        # Exclude invalid z values from mean reduction in case that there are any
        z = z[mask_valid]
        # Meters to milimeters
        z *= 1000.0

        depth = torch.zeros(self.shape_image, dtype=z.dtype, device=self.device)
        depth = depth.view(-1)
        depth.scatter_reduce_(dim=0, index=coords_uvflat, src=z, reduce="mean")
        # Lense has actually 190 degree FOV, so we need to handle negative depth
        depth[depth < 0] = 0.0

        if use_knn_interpolate:
            coords_uv_full = torch.stack(
                torch.meshgrid(
                    torch.arange(self.shape_image[0] // ratio_downsampling, device=self.device) * ratio_downsampling,
                    torch.arange(self.shape_image[1] // ratio_downsampling, device=self.device) * ratio_downsampling,
                ),
                dim=-1,
            )
            coords_uvflat_unique = torch.unique(coords_uvflat, sorted=False)
            # torch.unique(coords_uv, sorted=False, dim=0) is not working with Pytorch 2.0.0
            coords_uv_unique = torch.stack((coords_uvflat_unique // self.shape_image[1], coords_uvflat_unique % self.shape_image[1]), dim=-1)

            coords_uv_full_flat = coords_uv_full.view(-1, 2)

            depth_flat = depth[coords_uvflat_unique]
            depth_flat = depth_flat.view(-1, 1)

            depth = geometric_unpool.knn_interpolate(depth_flat.float(), coords_uv_unique.float(), coords_uv_full_flat.float(), k=1)
            depth = depth.view(-1, 1, self.shape_image[0] // ratio_downsampling, self.shape_image[1] // ratio_downsampling)
            depth = torch.nn.functional.upsample(depth, size=self.shape_image, mode="nearest", align_corners=None)
        else:
            depth = depth.view(-1, 1, self.shape_image[0], self.shape_image[1])

        return depth
