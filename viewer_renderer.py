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
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def color_map(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2 ,0, 1)
    return map

class ViewerRenderer:
    def __init__(
            self,
            gaussian_model,
            pipe,
            background_color,
            
    ):
        super().__init__()
        self.gaussian_model = gaussian_model
        self.pipe = pipe
        self.background_color = background_color

    def render_viewer(self,
                    viewpoint_camera, 
                    pc : GaussianModel, 
                    pipe, 
                    bg_color : torch.Tensor, 
                    scaling_modifier = 1.0, 
                    show_ptc: bool=False,
                    point_size = 0.001,
                    override_color = None, 
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
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        means3D = pc.get_xyz
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points
        opacity = pc.get_opacity
        opacity = pc.get_opacity

        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            # currently don't support normal consistency loss if use precomputed covariance
            splat2world = pc.get_covariance(scaling_modifier)
            W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
            near, far = viewpoint_camera.znear, viewpoint_camera.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, far-near, near],
                [0, 0, 0, 1]]).float().cuda().T
            world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
            cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
        
        pipe.convert_SHs_python = False
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        if show_ptc:
            scales = torch.full(scales.shape, point_size).to(scales.device)

        if valid_range is not None:
            is_x_in_range = (valid_range[0][0] <= means3D[:, 0]) & (means3D[:, 0] <= valid_range[0][1])
            is_y_in_range = (valid_range[1][0] <= means3D[:, 1]) & (means3D[:, 1] <= valid_range[1][1])
            is_z_in_range = (valid_range[2][0] <= means3D[:, 2]) & (means3D[:, 2] <= valid_range[2][1])
            is_in_box = is_x_in_range & is_y_in_range & is_z_in_range
        else:
            is_in_box = torch.ones(means3D.shape[0], dtype=torch.bool, device=means3D.device)

        rendered_image, radii, allmap = rasterizer(
            means3D = means3D[is_in_box],
            means2D = means2D[is_in_box],
            shs = shs[is_in_box],
            colors_precomp = colors_precomp,
            opacities = opacity[is_in_box],
            scales = scales[is_in_box],
            rotations = rotations[is_in_box],
            cov3D_precomp = cov3D_precomp
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
        surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()

        # fix normal truncation 
        render_normal = torch.nn.functional.normalize(render_normal, dim=0) * 0.5 + 0.5
        surf_normal = surf_normal * 0.5 + 0.5
        view_normal = -torch.nn.functional.normalize(allmap[2:5], dim=0) * 0.5 + 0.5

        rets ={'render': rendered_image,
                'rend_alpha': color_map(render_alpha.unsqueeze(dim=-1)), # render_alpha.repeat(3, 1, 1),
                'rend_normal': render_normal,
                'view_normal': view_normal,
                'surf_depth': color_map(surf_depth.unsqueeze(dim=-1)),
                'surf_normal': surf_normal,
                'rend_dist': color_map(render_dist.unsqueeze(dim=-1))
        }
        return rets

    def get_outputs(self, 
                    camera, 
                    scaling_modifier: float=1., 
                    valid_range: tuple=None, 
                    split: bool=False, 
                    slider: float=0.5,
                    show_ptc: bool=False,
                    point_size: float=0.1,
                    render_type = "render",
                    render_type1 = "render", 
                    render_type2 = "render", 
                    ):
        def get_result(results, type):
            if type in results.keys():
                return results[type]
            elif type == 'curvature':
                return color_map(gradient_map(results['surf_normal']))
            elif type == 'edge':
                return color_map(gradient_map(results['render']))
            else:
                return results['render']
        results = self.render_viewer(camera, 
                        self.gaussian_model,
                        self.pipe,
                        self.background_color,
                        scaling_modifier, 
                        valid_range = valid_range,
                        show_ptc = show_ptc, 
                        point_size = point_size, 
                        )
        if not split: 
            return get_result(results, render_type)
        else:
            result = torch.zeros_like(results['render'])
            _, _, render_h = result.shape
            result[:, :, :int(render_h * slider)] = get_result(results, render_type1)[:, :, :int(render_h * slider)]
            result[:, :, int(render_h * slider):] = get_result(results, render_type2)[:, :, int(render_h * slider):]
            result[:, :, int(render_h * slider)] = torch.ones_like(result[:, :, int(render_h * slider)])

            return result
