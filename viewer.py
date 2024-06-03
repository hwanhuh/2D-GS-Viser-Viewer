import os
import math
import glob
import time
import json
import torch
import numpy as np
import viser
import viser.transforms as vtf

from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Literal, List
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from arguments import ModelParams, PipelineParams, get_combined_args

from viewer_model import GaussianModelforViewer as GaussianModel
from viewer_renderer import ViewerRenderer
from internal.viewer import ClientThread
from internal.viewer.ui import populate_render_tab, TransformPanel, EditPanel
from internal.viewer.ui.up_direction_folder import UpDirectionFolder

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"

class Viewer:
    def __init__(
            self,
            model_paths: str,
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
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
            white_background: bool=True,
            iterations: int=30000,
            from_direct_path: str = None, 
    ):
        self.device = torch.device("cuda")
        self.render_type_name = {
            "RGB": 'render', 
            "Alpha": 'rend_alpha', 
            "Normal": 'rend_normal', 
            "View-Normal": 'view_normal',
            "Depth": 'surf_depth',
            "Depth2Normal": 'surf_normal',
        }
        self.model_paths = model_paths[0]
        self.host = host
        self.port = port
        self.background_color = background_color
        self.image_format = image_format
        self.sh_degree = sh_degree
        self.enable_transform = enable_transform
        self.show_cameras = show_cameras

        self.up_direction = np.asarray([0., 0., 1.])
        self.camera_center = np.asarray([0., 0., 0.])
        self.default_camera_position = default_camera_position
        self.default_camera_look_at = default_camera_look_at

        self.simplified_model = True
        self.show_edit_panel = True
        self.show_render_panel = True
        self.show_render_panel = ~no_render_panel

        parser = ArgumentParser(description="Vieweer Parameters")
        self.pipe = PipelineParams(parser)
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # load and create models
        if not self.model_paths.lower().endswith('.ply'):
            self.ply_path = os.path.join(self.model_paths, "point_cloud", "iteration_" + str(iterations), "point_cloud.ply")
        else: self.ply_path = self.model_paths
        if not os.path.exists(self.ply_path):
            print(self.ply_path)
            raise FileNotFoundError
        print('[INFO] ply path loaded from: ', self.ply_path)

        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        self.gaussian_model.load_ply(self.ply_path)
        self.viewer_renderer = ViewerRenderer(self.gaussian_model, self.pipe, self.background)
        training_output_base_dir = self.model_paths

        # reorient the scene
        cameras_json_path = cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient)
        if up is not None:
            self.camera_transform = torch.eye(4, dtype=torch.float)
            up = torch.tensor(up)
            up = -up / torch.linalg.norm(up)
            self.up_direction = up.numpy()

        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)
        if len(self.camera_poses) > 0:
            self.camera_center = np.mean(np.asarray([i["position"] for i in self.camera_poses]), axis=0)

        self.loaded_model_count = 1
        self.clients = {}

    def _reorient(self, cameras_json_path: str, mode: str):
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # detect whether cameras.json exists
        is_cameras_json_exists = os.path.exists(cameras_json_path)

        if is_cameras_json_exists is False:
            if mode == "enable":
                raise RuntimeError("{} not exists".format(cameras_json_path))
            else:
                return transform

        print("load {}".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))
        self.up_direction = up.numpy()

        return transform

    def load_camera_poses(self, cameras_json_path: str):
        if os.path.exists(cameras_json_path) is False:
            return []
        with open(cameras_json_path, "r") as f:
            return json.load(f)

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
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(205, 25, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = True

        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

        # def update_camera_scale(_):
        #     with viser_server.atomic():
        #         for i in self.camera_handles:
        #             i.scale = self.camera_scale_slider.value

        with viser_server.add_gui_folder("Cameras"):
            self.toggle_camera_button = viser_server.add_gui_button("Toggle Camera Visibility")
            # self.camera_scale_slider = viser_server.add_gui_slider(
            #     "Camera Scale",
            #     min=0.,
            #     max=1.,
            #     step=0.01,
            #     initial_value=0.1,
            # )
        self.toggle_camera_button.on_click(toggle_camera_visibility)
        # self.camera_scale_slider.on_update(update_camera_scale)
    
    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)

        buttons = (
            TitlebarButton(
                text="Simple Viser Viewer for 2D Gaussian Splatting",
                icon="GitHub",
                href="https://github.com/hwanhuh/2D-GS-Viser-Viewer/tree/main",
            ),
            TitlebarButton(
                text="Hwan Heo",
                icon="Keyboard",
                href="https://www.linkedin.com/in/hwan-heo-0905korea",
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

        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.add_gui_tab_group()
        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        with tabs.add_tab("General"):
            with server.add_gui_folder("Render Options"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.add_gui_slider(
                    "Max Res when Moving",
                    min=128,
                    max=1920,
                    step=128,
                    initial_value=1024,
                )
                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

                self.render_type = server.add_gui_dropdown(
                    "Render Type", ("RGB", "Alpha", "Normal", "View-Normal", "Depth", "Depth2Normal")
                )
                self.render_type.on_update(self._handle_render_type_updated)

            with server.add_gui_folder("Split Mode"):
                self.enable_split = server.add_gui_checkbox(
                    "Split Mode",
                    initial_value=False,
                )
                self.mode_slider = server.add_gui_slider(
                    "Split Slider",
                    min=0.,
                    max=0.99,
                    step=0.01,
                    initial_value=0.5,
                )
                self.mode_slider.on_update(self._handle_option_updated)

                self.render_type1 = server.add_gui_dropdown(
                    "Mode 1", ("RGB", "Alpha", "Normal", "View-Normal", "Depth", "Depth2Normal")
                )
                self.render_type2 = server.add_gui_dropdown(
                    "Mode 2", ("RGB", "Alpha", "Normal", "View-Normal", "Depth", "Depth2Normal")
                )
                self.render_type1.on_update(self._handle_render_type1_updated)
                self.render_type2.on_update(self._handle_render_type2_updated)

            with server.add_gui_folder("Gaussian Model"):
                self.scaling_modifier = server.add_gui_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=1.,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                if self.viewer_renderer.gaussian_model.max_sh_degree > 0:
                    self.active_sh_degree_slider = server.add_gui_slider(
                        "Active SH Degree",
                        min=0,
                        max=self.viewer_renderer.gaussian_model.max_sh_degree,
                        step=1,
                        initial_value=self.viewer_renderer.gaussian_model.max_sh_degree,
                    )
                    self.active_sh_degree_slider.on_update(self._handle_activate_sh_degree_slider_updated)

                self.enable_ptc = server.add_gui_checkbox(
                    "as Pointcloud",
                    initial_value=False,
                )
                self.point_size = server.add_gui_number(
                    "Point Size",
                    min=0.001,
                    initial_value=0.001,
                )
                self.point_size.on_update(self._handle_option_updated)

            with server.add_gui_folder("Crop Box"):
                self.enable_crop = server.add_gui_checkbox(
                    "use Crop Box",
                    initial_value=False,
                )
                self.x_min = server.add_gui_slider(
                    "x_min",
                    min=-10.,
                    max=0.,
                    step=0.1,
                    initial_value=-3.,
                )
                self.x_max = server.add_gui_slider(
                    "x_max",
                    min=0.,
                    max=10.,
                    step=0.1,
                    initial_value=3.,
                )
                self.y_min = server.add_gui_slider(
                    "y_min",
                    min=-10.,
                    max=0.,
                    step=0.1,
                    initial_value=-3.,
                )
                self.y_max = server.add_gui_slider(
                    "y_max",
                    min=0.,
                    max=10.,
                    step=0.1,
                    initial_value=3.,
                )
                self.z_min = server.add_gui_slider(
                    "z_min",
                    min=-10.,
                    max=0.,
                    step=0.1,
                    initial_value=-3.,
                )
                self.z_max = server.add_gui_slider(
                    "z_max",
                    min=0.,
                    max=10.,
                    step=0.1,
                    initial_value=3.,
                )
                self.x_min.on_update(self._handle_option_updated)
                self.x_max.on_update(self._handle_option_updated)
                self.y_min.on_update(self._handle_option_updated)
                self.y_max.on_update(self._handle_option_updated)
                self.z_min.on_update(self._handle_option_updated)
                self.z_max.on_update(self._handle_option_updated)

            # add cameras
            if self.show_cameras is True:
                self.add_cameras_to_scene(server)

            go_to_scene_center = server.add_gui_button(
                "Go to scene center",
            )

            @go_to_scene_center.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.position = self.camera_center + np.asarray([2., 0., 0.])
                event.client.camera.look_at = self.camera_center

        if self.show_edit_panel is True:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)

        self.transform_panel: TransformPanel = None
        if self.enable_transform is True:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self, self.loaded_model_count)

        if self.show_render_panel is True:
            with tabs.add_tab("Render"):
                populate_render_tab(
                    server,
                    self,
                    self.model_paths,
                    Path("./"),
                    orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                    enable_transform=self.enable_transform,
                    background_color=self.background_color,
                    sh_degree=self.sh_degree,
                )

        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        if block is True:
            while True:
                time.sleep(999)

    def _handle_render_type_updated(self, _):
        if self.render_type.value in self.render_type_name.keys():
            self.viewer_renderer.render_type = self.render_type_name[self.render_type.value]
        else:
            self.viewer_renderer.render_type = self.render_type_name['RGB']
        self._handle_option_updated(_)

    def _handle_render_type1_updated(self, _):
        if self.render_type.value in self.render_type_name.keys():
            self.viewer_renderer.render_type1 = self.render_type_name[self.render_type1.value]
        else:
            self.viewer_renderer.render_type1 = self.render_type_name['RGB']
        self._handle_option_updated(_)

    def _handle_render_type2_updated(self, _):
        if self.render_type.value in self.render_type_name.keys():
            self.viewer_renderer.render_type2 = self.render_type_name[self.render_type2.value]
        else:
            self.viewer_renderer.render_type2 = self.render_type_name['RGB']
        self._handle_option_updated(_)

    def _handle_activate_sh_degree_slider_updated(self, _):
        self.viewer_renderer.gaussian_model.active_sh_degree = self.active_sh_degree_slider.value
        self._handle_option_updated(_)

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass

    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)

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
    # define arguments
    parser = ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
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

    parser.add_argument("--white_background", action="store_true", default=False)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--float32_matmul_precision", "--fp", type=str, default=None)
    parser.add_argument("--from_direct_path", type=str, default=None)
    args = parser.parse_args()

    # set torch float32_matmul_precision
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    del args.float32_matmul_precision

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)
    viewer.start()
