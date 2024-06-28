import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    
    # gradient magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2).norm(dim=0, keepdim=True)
    return magnitude

class ViewerRenderer:
    def __init__(self,
                gaussian_model,
                background_color, 
                do_initialize=True):
        super().__init__()
        self.gaussian_model = gaussian_model
        self.background_color = background_color
        self.clm_colors = torch.tensor(plt.cm.get_cmap("turbo").colors, device="cuda")
        if do_initialize:
            self.update_pc_features()

    def update_pc_features(self):
        self.means3D = self.gaussian_model.get_xyz
        self.all_ids = torch.ones(self.means3D.shape[0], dtype=torch.bool, device=self.means3D.device)
        screenspace_points = torch.zeros_like(self.means3D, dtype=self.means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        self.means2D = screenspace_points
        self.opacity = self.gaussian_model.get_opacity
        self.scales = self.gaussian_model.get_scaling
        self.rotations = self.gaussian_model.get_rotation
        self.shs = self.gaussian_model.get_features

    def disk_kernel(self, opacity):
        return torch.exp(-1/2 * 100 * torch.clamp(opacity-0.5, min=0) ** 2)

    def color_map(self, map):
        if not map.min() == map.max():
            map = (map - map.min()) / (map.max() - map.min())
            map = (map * 255).round().long().squeeze()
            map = self.clm_colors[map].permute(2, 0, 1)
            return map
        else:
            map = torch.zeros_like(map, device=map.device).round().long().squeeze()
            map = self.clm_colors[map].permute(2, 0, 1)
            return map

    def render_viewer(self,
                    viewpoint_camera, 
                    active_sh_degree, 
                    scaling_modifier, 
                    depth_ratio,
                    bg_color : torch.Tensor, 
                    sparsity: int = 1,
                    show_ptc: bool = False,
                    show_disk: bool = False,
                    point_size: float = 0.001,
                    valid_range = None):
        """
        Render the scene. 
        Background tensor (bg_color) must be on GPU!
        """
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1., #self.gaussian_model.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree, #self.gaussian_model.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if valid_range is not None:
            is_x_in_range = (valid_range[0][0] <= self.means3D[:, 0]) & (self.means3D[:, 0] <= valid_range[0][1])
            is_y_in_range = (valid_range[1][0] <= self.means3D[:, 1]) & (self.means3D[:, 1] <= valid_range[1][1])
            is_z_in_range = (valid_range[2][0] <= self.means3D[:, 2]) & (self.means3D[:, 2] <= valid_range[2][1])
            is_in_box = is_x_in_range & is_y_in_range & is_z_in_range
        else:
            is_in_box = self.all_ids

        rendered_image, radii, allmap = rasterizer(
            means3D = self.means3D[is_in_box][::sparsity],
            means2D = self.means2D[is_in_box][::sparsity],
            shs = self.shs[is_in_box][::sparsity],
            colors_precomp = None,
            opacities = self.disk_kernel(self.opacity[is_in_box][::sparsity]) if show_disk else self.opacity[is_in_box][::sparsity],
            scales = scaling_modifier * (torch.full(self.scales.shape, point_size*0.1).to(self.scales.device)[is_in_box][::sparsity] if show_ptc else self.scales[is_in_box][::sparsity]),
            rotations = self.rotations[is_in_box][::sparsity],
            cov3D_precomp = None
        )
        
        # get normal map & transform normal from view space to world space
        render_alpha = allmap[1:2]
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map & depth map & depth-to-normal map 
        render_dist = allmap[6:7]
        surf_depth = render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()

        # fix normal truncation 
        render_normal = torch.nn.functional.normalize(render_normal, dim=0) * 0.5 + 0.5
        surf_normal = surf_normal * 0.5 + 0.5
        view_normal = -torch.nn.functional.normalize(allmap[2:5], dim=0) * 0.5 + 0.5

        rets ={'render': rendered_image,
                'rend_alpha': self.color_map(render_alpha.unsqueeze(dim=-1)), # render_alpha.repeat(3, 1, 1),
                'rend_normal': render_normal,
                'view_normal': view_normal,
                'surf_depth': self.color_map(surf_depth.unsqueeze(dim=-1)),
                'surf_normal': surf_normal,
                'rend_dist': self.color_map(render_dist.unsqueeze(dim=-1))
        }
        return rets

    def get_outputs(self, 
                    camera, 
                    valid_range: tuple=None, 
                    split: bool=False, 
                    slider: float=0.5,
                    show_ptc: bool=False,
                    show_disk: bool=False,
                    point_size: float=0.01,
                    active_sh_degree: int=3, 
                    scaling_modifier: float=1., 
                    sparsity: int=1,
                    depth_ratio: float=0.,
                    render_type: str="render",
                    render_type1: str="render", 
                    render_type2: str="render", 
                    ):
        def get_result(results, type):
            if type in results.keys():
                return results[type]
            elif type == 'curvature':
                return self.color_map(gradient_map(results['surf_normal']))
            elif type == 'edge':
                return self.color_map(gradient_map(results['render']))
            else:
                # handle exception as RGB render
                return results['render']

        results = self.render_viewer(camera, 
                                    active_sh_degree, 
                                    scaling_modifier, 
                                    depth_ratio,
                                    self.background_color,
                                    sparsity = sparsity,
                                    valid_range = valid_range,
                                    show_ptc = show_ptc, 
                                    show_disk = show_disk,
                                    point_size = point_size, 
                                    )
        if not split: 
            return get_result(results, render_type)
        else:
            result = torch.zeros_like(results['render'])
            _, _, render_h = result.shape
            slider_pos = int(render_h * slider)
            result[:, :, :slider_pos] = get_result(results, render_type1)[:, :, :slider_pos]
            result[:, :, slider_pos:] = get_result(results, render_type2)[:, :, slider_pos:]
            result[:, :, slider_pos] = torch.ones_like(result[:, :, slider_pos])

            return result