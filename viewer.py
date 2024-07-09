import os
import sys 
import math
import glob
import time
import json
import torch
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import threading
import warnings 
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '2d_gaussian_splatting'))

from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Literal, List
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from arguments import ModelParams, PipelineParams, get_combined_args
from internal.viewer import ViewerRenderer, ClientThread
from internal.viewer import GaussianModelforViewer as GaussianModel
from internal.viewer.ui import RenderPanel, TransformPanel, EditPanel

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"

class Viewer:
    def __init__(
            self,
            args, 
            model_paths: str,
            source_path: str = '',
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0.5, 0.5, 0.5),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 3,
            enable_transform: bool = False,
            show_cameras: bool = False,
            cameras_json: str = None,
            up: list = None,
            default_camera_position: List = None,
            default_camera_look_at: List = None,
            no_edit_panel: bool = False,
            no_render_panel: bool = False,
            iterations: int=30000,
            crop_box_size: float=16.0,
            from_direct_path: str = None, 
            is_training: bool = False,
    ):
        self.render_type_name = {
            "RGB": 'render', 
            "Edge": 'edge',
            "Alpha": 'rend_alpha', 
            "Normal": 'rend_normal', 
            "View-Normal": 'view_normal',
            "Depth": 'surf_depth',
            "Depth-Distort": 'rend_dist',
            "Depth-to-Normal": 'surf_normal',
            "Depth-to-Curvature": 'curvature',
            "None": 'render',
        }
        self.args = args 
        self.model_paths = model_paths[0]
        self.source_path = source_path
        self.cameras_json = os.path.join(self.model_paths, "cameras.json") if cameras_json is None else cameras_json

        self.host = host
        self.port = port
        self.background_color = torch.tensor(background_color, dtype=torch.float32, device="cuda")
        self.image_format = image_format
        self.sh_degree = sh_degree
        self.enable_transform = enable_transform
        self.show_cameras = show_cameras
        self.crop_box_size = crop_box_size

        self.device = torch.device("cuda")
        self.total_device_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2

        self.up_direction = np.asarray([0., 0., 1.])
        self.camera_center = np.asarray([0., 0., 0.])
        self.default_camera_position = default_camera_position
        self.default_camera_look_at = default_camera_look_at
        self.is_training = is_training
        self.show_edit_panel = ~no_edit_panel
        self.show_render_panel = ~no_render_panel

        # init model & scene 
        self._init_models(iterations)
        self._init_scene_camera_transform(self.cameras_json, reorient, up)
        self._init_camera_poses(self.cameras_json)
        self.clients = {}
        
    def _init_models(self, iterations):
        # init gaussian model & renderer
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        if not self.is_training:
            self.iteration = iterations
            self.ply_path = os.path.join(self.model_paths, "point_cloud", f"iteration_{iterations}", "point_cloud.ply") if not self.model_paths.lower().endswith('.ply') else self.model_paths
            if not os.path.exists(self.ply_path):
                print(f'[Alert] there is no pointcloud in: {self.ply_path}')
                raise FileNotFoundError
            print(f'[INFO] ply path loaded from: {self.ply_path}')
            self.gaussian_model.load_ply(self.ply_path)
            print(f'[INFO] number of points: {self.gaussian_model._xyz.shape[0]}')
        self.viewer_renderer = ViewerRenderer(self.gaussian_model, self.background_color, not self.is_training)

    def _init_scene_camera_transform(self, cameras_json_path, mode, up):
        transform = torch.eye(4, dtype=torch.float)
        self.camera_transform = transform
        if mode == "disable" or not os.path.exists(cameras_json_path): return
        
        print(f"[Info] Load cameras from: {cameras_json_path}")
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up_vector = torch.zeros(3)
        for cam in cameras:
            up_vector += torch.tensor(cam["rotation"])[:3, 1]
        up_vector = -up_vector / torch.linalg.norm(up_vector)
        print(f"[INFO] up vector = {up_vector}")
        self.up_direction = up_vector.numpy()

        if up is not None:
            transform = torch.eye(4, dtype=torch.float)
            up_vector = torch.tensor(up)
            up_vector = -up_vector / torch.linalg.norm(up_vector)
            self.up_direction = up_vector.numpy()

        self.camera_transform = transform

    def _init_camera_poses(self, cameras_json_path):
        if not os.path.exists(cameras_json_path):
            return []
        with open(cameras_json_path, "r") as f:
            camera_poses = json.load(f)
        if camera_poses:
            self.camera_center = np.mean(np.asarray([i["position"] for i in camera_poses]), axis=0)
        self.camera_poses = camera_poses

    def _get_training_gaussians(self, new_gaussians):
        # slow and large gpu consumption
        self.gaussian_model._xyz = new_gaussians._xyz.clone().detach()
        self.gaussian_model._scaling = new_gaussians._scaling.clone().detach()
        self.gaussian_model._opacity = new_gaussians._opacity.clone().detach()
        self.gaussian_model._rotation = new_gaussians._rotation.clone().detach()
        self.gaussian_model._features_dc = new_gaussians._features_dc.clone().detach()
        self.gaussian_model._features_rest = new_gaussians._features_rest.clone().detach()
        self.viewer_renderer = ViewerRenderer(self.gaussian_model, self.background_color, self.is_training)

    def get_gpu_memory_usage(self):
        total_memory = torch.cuda.memory_allocated() + torch.cuda.memory_reserved() 
        return f"{total_memory / 1024 ** 2:.1f} / {self.total_device_memory:.1f} MB"

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []
        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for camera in self.camera_poses:
            name = camera["img_name"]
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.05,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(255, 255, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.show_cameras_frustrum = viser_server.add_gui_button("Show Train Cameras")
        self.camera_visible = True
        @self.show_cameras_frustrum.on_click
        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        self._setup_titles(server)
        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.add_gui_tab_group()
        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        # setup panels 
        self._setup_general_features_folder(server, tabs)

        if self.show_edit_panel:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)
                @self.edit_panel.show_point_cloud_checkbox.on_update
                @self.edit_panel.show_mesh_button.on_click 
                @self.edit_panel.unshow_mesh_button.on_click
                def _(event): 
                    with server.atomic(): self._handle_option_updated(_)

        self.transform_panel: TransformPanel = None
        if self.enable_transform:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self)

        if self.show_render_panel:
            with tabs.add_tab("Render"):
                self.render_panel = RenderPanel(server, 
                                                self, 
                                                self.model_paths, 
                                                Path('./renders'),
                                                orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                                                enable_transform=self.enable_transform,
                                                background_color=self.background_color.detach().cpu().numpy().tolist(),
                                                sh_degree=self.sh_degree,)
                
        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)
        if block is True:
            while True:
                time.sleep(999)
    
    def _setup_titles(self, server):
        buttons = (
            TitlebarButton(
                text="Simple Viser Viewer for 2D Gaussian Splatting",
                icon="GitHub",
                href="https://github.com/hwanhuh/2D-GS-Viser-Viewer/tree/main",
            ),
            TitlebarButton(
                text="Hwan Heo",
                icon="GitHub",
                href="https://github.com/hwanhuh",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://viser.studio/latest/_static/logo.svg",
            image_alt="Logo",
            href="https://github.com/nerfstudio-project/viser"
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = server.add_gui_rgb("Brand color", (10, 10, 10), visible=False)
        server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )

    def _setup_general_features_folder(self, server, tabs):
        with tabs.add_tab("General"):
            if self.is_training:
                with server.add_gui_folder("Training Infos"):
                    self.iter = server.add_gui_text('Iteration', initial_value = '0')
                    self.loss = server.add_gui_text('Loss', initial_value = '0.0')
                    self.dist = server.add_gui_text('distortion', initial_value = '0.0')
                    self.norm = server.add_gui_text('normal', initial_value = '0.0')
                    self.gpu_mem = server.add_gui_text(
                        'Memory Usage',
                        initial_value = self.get_gpu_memory_usage()
                    )
                    self.fps = server.add_gui_text(
                        'fps',
                        initial_value = ' frame/sec'
                    )
            else:
                with server.add_gui_folder("Status"):
                    self.gpu_mem = server.add_gui_text(
                        'Memory Usage',
                        initial_value = self.get_gpu_memory_usage()
                    )
                    self.fps = server.add_gui_text(
                        'fps',
                        initial_value = ' frame/sec'
                    )
            with server.add_gui_folder("Image Options"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.max_res_when_moving = server.add_gui_slider(
                    "(when Move)",
                    min=128,
                    max=1920,
                    step=128,
                    initial_value=1024,
                )
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "(when Move)",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

            with server.add_gui_folder("Render Options"):
                self.render_type = server.add_gui_dropdown(
                    "Render Type", tuple(self.render_type_name.keys())[:-1]
                )
                self.depth_ratio_slider = server.add_gui_slider(
                    "Depth Ratio (mean ~ med)",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=0.,
                )
            
                with server.add_gui_folder("screen split"):
                    self.enable_split = server.add_gui_checkbox(
                        "use Split",
                        initial_value=False,
                    )
                    self.mode_slider = server.add_gui_slider(
                        "Split Slider",
                        min=0.,
                        max=0.99,
                        step=0.01,
                        initial_value=0.5,
                    )

                    self.render_type1 = server.add_gui_dropdown(
                        "Left Type", tuple(self.render_type_name.keys())[:-1]
                    )
                    self.render_type2 = server.add_gui_dropdown(
                        "Right Type", tuple(self.render_type_name.keys())[:-1]
                    )
            
            with server.add_gui_folder("Gaussian Model"):
                self.enable_ptc = server.add_gui_checkbox(
                    "as Pointcloud",
                    initial_value=False,
                )
                self.surfel_mode = server.add_gui_button_group("View Type", ("ptc", "disk"))
                self.point_size = server.add_gui_slider(
                    "Point Size",
                    min=0.001,
                    max=0.1,
                    initial_value=0.01,
                    step=0.001,
                )
                self.scale_slider = server.add_gui_slider(
                    "Scaler",
                    min=0.1,
                    max=2.,
                    step=0.1,
                    initial_value=1.,
                )
                self.sparsity_slider = server.add_gui_slider(
                    "Sparsity",
                    min=1,
                    max=10,
                    step=1,
                    initial_value=1,
                )
                
                if self.viewer_renderer.gaussian_model.max_sh_degree > 0:
                    self.active_sh_degree_slider = server.add_gui_slider(
                        "SH Degree",
                        min=0,
                        max=self.viewer_renderer.gaussian_model.max_sh_degree,
                        step=1,
                        initial_value=self.viewer_renderer.gaussian_model.max_sh_degree,
                    )
                    @self.active_sh_degree_slider.on_update
                    def _(event): 
                        with server.atomic(): self._handle_option_updated(_)

            with server.add_gui_folder("Crop Box"):
                self.enable_crop = server.add_gui_checkbox(
                    "use Crop",
                    initial_value=False,
                )
                self.box_x = server.add_gui_multi_slider(
                    'x range', 
                    min = -self.crop_box_size,
                    max = self.crop_box_size,
                    step = 0.1,
                    initial_value= [-4.0, 4.0]
                )
                self.box_y = server.add_gui_multi_slider(
                    'y range', 
                    min = -self.crop_box_size,
                    max = self.crop_box_size,
                    step = 0.1,
                    initial_value=[-4.0, 4.0]
                )
                self.box_z = server.add_gui_multi_slider(
                    'z range', 
                    min = -self.crop_box_size,
                    max = self.crop_box_size,
                    step = 0.1,
                    initial_value=[-4.0, 4.0]
                )

            # add cameras
            if self.show_cameras:
                self.add_cameras_to_scene(server)

            @self.render_type.on_update
            @self.render_type1.on_update
            @self.render_type2.on_update
            @self.enable_split.on_update
            @self.mode_slider.on_update
            @self.depth_ratio_slider.on_update
            @self.scale_slider.on_update
            @self.sparsity_slider.on_update
            @self.enable_ptc.on_update
            @self.surfel_mode.on_click
            @self.point_size.on_update
            @self.enable_crop.on_update
            @self.box_x.on_update 
            @self.box_y.on_update 
            @self.box_z.on_update
            def _(event): 
                with server.atomic(): self._handle_option_updated(_)

            go_to_scene_center = server.add_gui_button("Go to scene center",)
            @go_to_scene_center.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.position = self.camera_center + np.asarray([2.5, 0., 0.])
                event.client.camera.look_at = self.camera_center

    def rerender_for_all_client(self):
        for client_id in self.clients:
            try:
                # switch to low resolution mode first, then notify the client to render
                self.clients[client_id].state = "low"
                self.clients[client_id].render_trigger.set()
            except:
                # ignore errors
                pass

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        client_thread = ClientThread(self, self.viewer_renderer, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """

        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--source_path", "-s", type=str, default="")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "-b",
                        type=str, nargs="+", default=["gray"],
                        help="e.g.: white, gray, black, [0 0 0], [0.5 0.5 0.5], [1 1 1]")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=3)
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true")
    parser.add_argument("--cameras-json", "--cameras_json", type=str, default=None)
    parser.add_argument("--up", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_position", "--dcp", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_look_at", "--dcla", nargs=3, required=False, type=float, default=None)

    parser.add_argument("--no_edit_panel", action="store_true", default=False)
    parser.add_argument("--no_render_panel", action="store_true", default=False)

    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--crop_box_size", type=float, default=16.0)
    parser.add_argument("--float32_matmul_precision", "--fp", type=str, default=None)
    parser.add_argument("--from_direct_path", type=str, default=None)
    args, unknown_args = parser.parse_known_args()

    # set torch float32_matmul_precision
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    del args.float32_matmul_precision

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = [1., 1., 1.]
        elif args.background_color[0] == "black":
            args.background_color = [0., 0., 0.]
        else:
            args.background_color = [0.5, 0.5, 0.5]
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(args, **viewer_init_args)
    viewer.start()
