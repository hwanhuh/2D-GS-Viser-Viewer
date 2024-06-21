import torch
import numpy as np 

import internal.utils.gaussian_utils as gaussian_utils
from scene import GaussianModel

class GaussianModelforViewer(GaussianModel):
    def __init__(self, sh_degree : int):
        super().__init__(sh_degree)
        self._opacity_origin = None
        self.scaling_modifier = 1.
        self.depth_ratio = 0.

    def select(self, mask: torch.tensor):
        if self._opacity_origin is None:
            self._opacity_origin = torch.clone(self._opacity)  # make a backup
        else:
            self._opacity = torch.clone(self._opacity_origin)

        # self._opacity[mask] = 0. inplace error!
        new_opacity = self._opacity.clone()
        new_opacity[mask] = 0. 
        self._opacity = new_opacity

    def delete_gaussians(self, mask: torch.tensor):
        gaussians_to_be_preserved = torch.bitwise_not(mask).to(self._xyz.device)
        self._xyz = self._xyz[gaussians_to_be_preserved]
        self._scaling = self._scaling[gaussians_to_be_preserved]
        self._rotation = self._rotation[gaussians_to_be_preserved]

        if self._opacity_origin is not None:
            self._opacity = self._opacity_origin
            self._opacity_origin = None
        self._opacity = self._opacity[gaussians_to_be_preserved]

        self._features_dc = self._features_dc[gaussians_to_be_preserved]
        self._features_rest = self._features_rest[gaussians_to_be_preserved]
        self.backup()

    def backup(self):
        # large memory consumption 
        self.org_xyz = self._xyz 
        self.org_scaling = self._scaling
        self.org_rotation = self._rotation
        self.org_features_dc = self._features_dc 
        self.org_features_rest = self._features_rest


    def transform_with_vectors(self,
                                idx: int,
                                scale: float,
                                r_wxyz: np.ndarray,
                                t_xyz: np.ndarray,):
        xyz = self.org_xyz
        scaling = self.org_scaling
        rotation = self.org_rotation
        features = torch.cat((self.org_features_dc, self.org_features_rest), dim=1)
        
        # rescale
        xyz, scaling = gaussian_utils.GaussianTransformUtils.rescale(
            xyz,
            scaling,
            scale
        )
        # rotate
        xyz, rotation, new_features = gaussian_utils.GaussianTransformUtils.rotate_by_wxyz_quaternions(
            xyz=xyz,
            rotations=rotation,
            features=features,
            quaternions=torch.tensor(r_wxyz).to(xyz),
        )
        # translate
        xyz = gaussian_utils.GaussianTransformUtils.translation(xyz, *t_xyz.tolist())

        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._features_dc = new_features[:, 0, None, :]
        self._features_rest = new_features[:, 1:]
