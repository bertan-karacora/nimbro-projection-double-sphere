import torch


class SamplerColor:
    def __init__(self, color_invalid=(255, 87, 51)):
        self.color_invalid = color_invalid
        self.device = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def coords_to_continuous(self, coords_uv, shape_image, inplace=False):
        if not inplace:
            coords_uv = coords_uv.clone()

        coords_uv[:, 0, :] = 2.0 * coords_uv[:, 0, :] / shape_image[2] - 1.0
        coords_uv[:, 1, :] = 2.0 * coords_uv[:, 1, :] / shape_image[1] - 1.0

        return coords_uv

    def sample_colors(self, coords_uv, images):
        coords_uv = self.coords_to_continuous(coords_uv, shape_image=images.shape[1:], inplace=True)
        coords_uv = coords_uv.permute(0, 2, 1)

        # grid_sample not implemented for dtype byte
        colors = torch.nn.functional.grid_sample(input=images, grid=coords_uv[:, None, :, :], align_corners=True)
        colors = colors[:, :, 0, :]

        return colors

    @torch.inference_mode()
    def __call__(self, coords_uv, images, mask_valid=None, use_half_precision=True):
        """Samples rgb values from image given uv coordinates.
        Shape of coords_uv: (B, 2, N)
        Shape of images: (B, 3, H, W)
        Shape of mask_valid: (B, N)"""
        # Clone coords_uv before changing values
        coords_uv = coords_uv.clone()

        if use_half_precision:
            coords_uv = coords_uv.half()
            images = images.half()

        coords_uv = coords_uv.to(self.device)
        images = images.to(self.device)
        if mask_valid is not None:
            mask_valid = mask_valid.to(self.device)

        colors = self.sample_colors(coords_uv, images)

        if mask_valid is not None:
            colors[:, 0, :].masked_fill_(~mask_valid, self.color_invalid[0])
            colors[:, 1, :].masked_fill_(~mask_valid, self.color_invalid[1])
            colors[:, 2, :].masked_fill_(~mask_valid, self.color_invalid[2])

        return colors
