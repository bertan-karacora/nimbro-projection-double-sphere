import torch
import torch_geometric.nn.unpool as geometric_unpool


class SamplerDepth:
    def __init__(self, shape_image=None, factor_downsampling=8, k_knn=1, mode_interpolation="bilinear"):
        self.coords_uv_full_flat = None
        self.device = None
        self.k_knn = k_knn
        self.mode_interpolation = mode_interpolation
        self.factor_downsampling = factor_downsampling
        self._shape_image = None
        self.shape_image = shape_image

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def shape_image(self):
        return self._shape_image

    @shape_image.setter
    def shape_image(self, value):
        has_changed = self._shape_image != value

        self._shape_image = value

        if has_changed:
            self.update_coords_uv_full_flat()

    def update_coords_uv_full_flat(self):
        u_full_downsampled = torch.arange(self.shape_image[1] // self.factor_downsampling, device=self.device) * self.factor_downsampling
        v_full_downsampled = torch.arange(self.shape_image[2] // self.factor_downsampling, device=self.device) * self.factor_downsampling
        coords_uv_full = torch.stack(torch.meshgrid(u_full_downsampled, v_full_downsampled), dim=-1)
        self.coords_uv_full_flat = coords_uv_full.view(-1, 2)
        self.coords_uv_full_flat = self.coords_uv_full_flat.float()

    def flatten_uv_coords(self, coords_uv):
        # Round to integers
        coords_uv = coords_uv.long()
        coords_uvflat = coords_uv[1, :] * self.shape_image[2] + coords_uv[0, :]

        return coords_uvflat

    def sample_depth(self, coords_uvflat, points, mask_valid=None):
        z = points[2, :]
        # Important to apply mask before scatter_reduce
        if mask_valid is not None:
            z = z[mask_valid]

        depth = torch.zeros(self.shape_image[1:], dtype=z.dtype, device=self.device)
        depth_flat = depth.view(-1)
        depth_flat.scatter_reduce_(dim=0, index=coords_uvflat, src=z, include_self=False, reduce="mean")

        return depth_flat

    def interpolate_knn(self, coords_uvflat, depth_flat):
        coords_uvflat_unique = torch.unique(coords_uvflat, sorted=False)
        # Use this because torch.unique(coords_uv, sorted=False, dim=1) does not work with Pytorch 2.0.0
        # Need uv stacked in dimension 1 for knn_interpolate
        coords_uv_unique = torch.stack((coords_uvflat_unique // self.shape_image[2], coords_uvflat_unique % self.shape_image[2]), dim=1)

        depth_flat = depth_flat[coords_uvflat_unique]
        # Need this dimension for knn_interpolate
        depth_flat = depth_flat[:, None]

        # Need float for input in knn_interpolate. Half apparently leads to artifacts/overflows
        depth_flat = geometric_unpool.knn_interpolate(depth_flat.float(), coords_uv_unique.float(), self.coords_uv_full_flat.float(), k=self.k_knn)
        depth = depth_flat.view(1, 1, self.shape_image[1] // self.factor_downsampling, self.shape_image[2] // self.factor_downsampling)
        depth = torch.nn.functional.interpolate(depth, size=self.shape_image[1:], mode=self.mode_interpolation, align_corners=True if self.mode_interpolation not in ["nearest"] else None)

        return depth

    @torch.inference_mode()
    def __call__(self, coords_uv, points, mask_valid=None, use_knn_interpolation=True):
        """Samples depth values from points given uv coordinates.
        Note: Currently just using first batch.
        Shape of coords_uv: (B, 2, N)
        Shape of points: (B, 3, N)
        Shape of mask_valid: (B, N)"""
        # Some operations are not working with batches
        coords_uv = coords_uv[0]
        points = points[0]
        mask_valid = mask_valid[0]

        coords_uv = coords_uv.to(self.device)
        points = points.to(self.device)
        if mask_valid is not None:
            mask_valid = mask_valid.to(self.device)

        if mask_valid is not None:
            coords_uv = coords_uv[:, mask_valid]

        coords_uvflat = self.flatten_uv_coords(coords_uv)
        depth_flat = self.sample_depth(coords_uvflat, points, mask_valid)

        if use_knn_interpolation:
            depth = self.interpolate_knn(coords_uvflat, depth_flat)
        else:
            depth = depth_flat.view(1, 1, *self.shape_image[1:])

        # Lense has actually 190 degree FOV, so we need to handle negative depth
        depth[depth < 0] = 0.0

        return depth
