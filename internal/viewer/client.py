import time
import threading
import traceback
import numpy as np
import torch
import viser
import json
import viser.transforms as vtf
from internal.cameras.cameras import Cameras
from internal.utils.graphics_utils import fov2focal

def load_camera_paths(self, json_file):
    with open(json_file, 'r') as file:
        camera_paths = json.load(file)
    return camera_paths

class ClientThread(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client
        self.render_trigger = threading.Event()
        self.last_move_time = 0
        self.last_camera = None  # store camera information
        self.state = "low"  # low or high render resolution
        self.stop_client = False  # whether stop this thread
        self.playing_preview = False # flag to indicate if we are playing a preview


        if viewer.default_camera_position is not None:
            client.camera.position = np.asarray(viewer.default_camera_position)
        if viewer.default_camera_look_at is not None:
            client.camera.look_at = np.asarray(viewer.default_camera_look_at)
        client.camera.up_direction = viewer.up_direction

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()
        if hasattr(viewer, 'render_panel'):
            @viewer.render_panel.preview_button.on_click
            def _(_) -> None:
                with self.client.atomic():
                    self.render_trigger.set()

    def send_camera_path(self, camera_paths, fps=30):
        for path in camera_paths:
            with self.client.atomic():
                R = vtf.SO3(wxyz=path["wxyz"])
                R = R @ vtf.SO3.from_x_radians(np.pi)
                R = torch.tensor(R.as_matrix())
                pos = torch.tensor(path["position"], dtype=torch.float64)
                c2w = torch.eye(4)
                c2w[:3, :3] = R
                c2w[:3, 3] = pos

                c2w = torch.matmul(self.viewer.camera_transform, c2w)
                c2w[:3, 1:3] *= -1

                w2c = torch.linalg.inv(c2w)
                R = w2c[:3, :3]
                T = w2c[:3, 3]

                fov = self.last_camera.fov
                aspect_ratio = path["aspect"]
                # Calculate resolution
                max_res, jpeg_quality = self.get_render_options()
                image_height = max_res
                image_width = int(image_height * aspect_ratio)
                if image_width > max_res:
                    image_width = max_res
                    image_height = int(image_width / aspect_ratio)
                    
                # Construct camera
                fx = torch.tensor([fov2focal(fov, max_res)], dtype=torch.float)
                camera = Cameras(
                    R=R.unsqueeze(0),
                    T=T.unsqueeze(0),
                    fx=fx,
                    fy=fx,
                    cx=torch.tensor([(image_width // 2)], dtype=torch.int),
                    cy=torch.tensor([(image_height // 2)], dtype=torch.int),
                    width=torch.tensor([image_width], dtype=torch.int),
                    height=torch.tensor([image_height], dtype=torch.int),
                    appearance_id=torch.tensor([0], dtype=torch.int),
                    normalized_appearance_id=torch.tensor([0.], dtype=torch.float),
                    time=torch.tensor([0], dtype=torch.float),
                    distortion_params=None,
                    camera_type=torch.tensor([0], dtype=torch.int),
                )[0].to_device(self.viewer.device)
                
                with torch.no_grad():
                    self.viewer.gpu_mem.value = self.viewer.get_gpu_memory_usage()
                    valid_range = None
                    if self.viewer.enable_crop.value:
                        valid_range = (self.viewer.box_x.value, self.viewer.box_y.value, self.viewer.box_z.value)
                    image = self.renderer.get_outputs(camera, 
                                                    scaling_modifier = self.viewer.scaling_modifier.value, 
                                                    valid_range = valid_range, 
                                                    split = self.viewer.enable_split.value, 
                                                    slider = self.viewer.mode_slider.value,
                                                    show_ptc = self.viewer.enable_ptc.value,
                                                    point_size = self.viewer.point_size.value,)
                    image = torch.clamp(image, max=1.)
                    image = torch.permute(image, (1, 2, 0))
                    self.client.set_background_image(
                        image.cpu().numpy(),
                        format=self.viewer.image_format,
                        jpeg_quality=jpeg_quality,
                    )
                self.render_trigger.set()
                time.sleep(1 / (fps*6))

    def render_and_send(self):
        if hasattr(self.viewer, 'render_panel'):
            if self.viewer.render_panel.play_preview:
                if not self.playing_preview and self.viewer.render_panel.preview_cameras is not None:
                    self.playing_preview = True
                    fps = self.viewer.render_panel.preview_cameras['fps']
                    self.send_camera_path(self.viewer.render_panel.preview_cameras['camera_path'], fps)
                    self.playing_preview = False
                    self.viewer.render_panel.play_preview = False
                    self.render_trigger.set()
                    return
        with self.client.atomic():
            cam = self.last_camera
            self.last_move_time = time.time()

            # get camera pose
            R = vtf.SO3(wxyz=self.client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(np.pi)
            R = torch.tensor(R.as_matrix())
            pos = torch.tensor(self.client.camera.position, dtype=torch.float64)
            c2w = torch.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = pos

            c2w = torch.matmul(self.viewer.camera_transform, c2w)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            # calculate resolution
            aspect_ratio = cam.aspect
            max_res, jpeg_quality = self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)

            # construct camera
            appearance_id = (0, 0.)
            fx = torch.tensor([fov2focal(cam.fov, max_res)], dtype=torch.float)
            camera = Cameras(
                R=R.unsqueeze(0),
                T=T.unsqueeze(0),
                fx=fx,
                fy=fx,
                cx=torch.tensor([(image_width // 2)], dtype=torch.int),
                cy=torch.tensor([(image_height // 2)], dtype=torch.int),
                width=torch.tensor([image_width], dtype=torch.int),
                height=torch.tensor([image_height], dtype=torch.int),
                appearance_id=torch.tensor([appearance_id[0]], dtype=torch.int),
                normalized_appearance_id=torch.tensor([appearance_id[1]], dtype=torch.float),
                time=torch.tensor([0], dtype=torch.float),
                distortion_params=None,
                camera_type=torch.tensor([0], dtype=torch.int),
            )[0].to_device(self.viewer.device)

            # if self.viewer.client_debugger.value:
            #     import pdb; pdb.set_trace()

            with torch.no_grad():
                self.viewer.gpu_mem.value = self.viewer.get_gpu_memory_usage()
                def get_valid_range():
                    if self.viewer.enable_crop.value:
                        return (self.viewer.box_x.value, self.viewer.box_y.value, self.viewer.box_z.value)
                    return None

                def render_image(camera):
                    valid_range = get_valid_range()
                    return self.renderer.get_outputs(
                        camera,
                        scaling_modifier=self.viewer.scaling_modifier.value,
                        valid_range=valid_range,
                        split=self.viewer.enable_split.value,
                        slider=self.viewer.mode_slider.value,
                        show_ptc=self.viewer.enable_ptc.value,
                        point_size=self.viewer.point_size.value,
                    )
                if hasattr(self.viewer, 'edit_panel') and self.viewer.edit_panel.show_point_cloud_checkbox.value:
                    image = self.viewer.background_color.unsqueeze(dim=1).unsqueeze(dim=2).expand([3, image_height, image_width])
                else:
                    image = render_image(camera)
                    image = torch.clamp(image, max=1.)

                image = torch.permute(image, (1, 2, 0))
                self.client.set_background_image(
                    image.cpu().numpy(),
                    format=self.viewer.image_format,
                    jpeg_quality=jpeg_quality,
                )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue

                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()

            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True

    def _destroy(self):
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None
