from dataclasses import dataclass
import numpy as np
import math
import viser
import viser.transforms as vst
from internal.utils.gaussian_utils import GaussianTransformUtils
from internal.viewer.ui.up_direction_folder import UpDirectionFolder


@dataclass
class ModelPose:
    wxyz: np.ndarray
    position: np.ndarray

    def copy(self):
        return ModelPose(
            wxyz=self.wxyz.copy(),
            position=self.position.copy(),
        )

    def to_dict(self):
        return {
            "wxyz": self.wxyz.tolist(),
            "position": self.position.tolist(),
        }


class TransformPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            i=0,
    ):
        self.server = server
        self.viewer = viewer

        self.transform_control_no_handle_update = False

        self.model_poses = []
        self.model_transform_controls: dict[int, viser.TransformControlsHandle] = {}
        self.model_size_sliders = []
        self.model_show_transform_control_checkboxes = []
        self.model_t_xyz_text_handle = []
        self.model_r_xyz_text_handle = []
        self.prev_t_xyz = (0., 0., 0.)
        self.prev_r_xyz = (0., 0., 0.)

        # for transforming, backup original position & rotations 
        self.viewer.gaussian_model.backup()

        # create gui folder for each model
        
        with server.add_gui_folder("Model Transform"):
            self.model_show_transform_control_checkbox = server.add_gui_checkbox(
                "use Transform",
                initial_value=False,
            )
            self._make_show_transform_control_checkbox_callback(i, self.model_show_transform_control_checkbox)
            self.model_show_transform_control_checkboxes.append(self.model_show_transform_control_checkbox)

            self.pose_control_size = server.add_gui_slider(
                "Control Mode",
                min=0.,
                max=10.,
                step=0.01,
                initial_value=1.0,
            )
            self.pose_control_size.on_update(self._update_pose_control_size)

            # model size control
            size_slider = server.add_gui_number(
                "Splat Size",
                min=0.,
                step=0.01,
                initial_value=1.,
            )
            self._make_size_slider_callback(i, size_slider)
            self.model_size_sliders.append(size_slider)

            # model pose control
            self.model_poses.append(ModelPose(
                np.asarray([1., 0., 0., 0.]),
                np.zeros((3,)),
            ))
            
            # add text input (synchronize with model pose control) that control model pose more precisely
            t_xyz_text_handle = server.add_gui_vector3(
                "Translation",
                initial_value=(0., 0., 0.),
                step=0.01,
            )
            self._make_t_xyz_text_callback(i, t_xyz_text_handle)
            self.model_t_xyz_text_handle.append(t_xyz_text_handle)

            r_xyz_text_handle = server.add_gui_vector3(
                "Rotation",
                initial_value=(0., 0., 0.),
                step=0.01,
            )
            self._make_r_xyz_text_callback(i, r_xyz_text_handle)
            self.model_r_xyz_text_handle.append(r_xyz_text_handle)

            self.set_to_default = server.add_gui_button("Set to default")
            @self.set_to_default.on_click
            def _(event: viser.GuiEvent) -> None:
                if self.model_show_transform_control_checkbox.value:
                    assert event.client is not None
                    self.viewer.gaussian_model.backup() # set current value as a defalt 
                    self.viewer.camera_center = self.viewer.camera_center + np.array(t_xyz_text_handle.value)
                    self.model_transform_controls[i].remove()
                    del self.model_transform_controls[i]
                    self.model_poses[i] = ModelPose(np.asarray([1., 0., 0., 0.]), np.zeros((3,)))
                    t_xyz_text_handle.value = (0., 0., 0.)
                    r_xyz_text_handle.value = (0., 0., 0.)
                    self._show_model_transform_handle(i)
        UpDirectionFolder(viewer, server)

        UpDirectionFolder(viewer, server)

        UpDirectionFolder(viewer, server)

    def _make_size_slider_callback(
            self,
            idx: int,
            slider: viser.GuiInputHandle,
    ):
        @slider.on_update
        def _(event: viser.GuiEvent) -> None:
            with self.server.atomic():
                self._transform_model(idx)
                self.viewer.rerender_for_client(event.client_id)

    def set_model_transform_control_value(self, idx, wxyz: np.ndarray, position: np.ndarray):
        if idx in self.model_transform_controls:
            self.transform_control_no_handle_update = True
            try:
                self.model_transform_controls[idx].wxyz = wxyz
                self.model_transform_controls[idx].position = position
            finally:
                self.transform_control_no_handle_update = False

    def _make_transform_controls_callback(
            self,
            idx,
            controls: viser.TransformControlsHandle,
    ) -> None:
        @controls.on_update
        def _(event: viser.GuiEvent) -> None:
            if self.transform_control_no_handle_update is True:
                return
            model_pose = self.model_poses[idx]
            model_pose.wxyz = controls.wxyz
            model_pose.position = controls.position

            self.model_t_xyz_text_handle[idx].value = model_pose.position.tolist()
            self.model_r_xyz_text_handle[idx].value = self.quaternion_to_euler_angle_vectorized2(model_pose.wxyz)

            self._transform_model(idx)
            self.viewer.rerender_for_all_client()

    def _show_model_transform_handle(
            self,
            idx: int,
    ):
        model_pose = self.model_poses[idx]
        controls = self.server.add_transform_controls(
            f"/model_transform/{idx}",
            scale=self.pose_control_size.value,
            wxyz=model_pose.wxyz,
            position=model_pose.position,
        )
        self._make_transform_controls_callback(idx, controls)
        self.model_transform_controls[idx] = controls

    def _make_show_transform_control_checkbox_callback(
            self,
            idx: int,
            checkbox: viser.GuiInputHandle,
    ):
        @checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            if checkbox.value is True:
                self._show_model_transform_handle(idx)
            else:
                if idx in self.model_transform_controls:
                    self.model_transform_controls[idx].remove()
                    del self.model_transform_controls[idx]

    def _update_pose_control_size(self, _):
        with self.server.atomic():
            for i in self.model_transform_controls:
                self.model_transform_controls[i].remove()
                self._show_model_transform_handle(i)

    def _transform_model(self, idx):
        model_pose = self.model_poses[idx]
        self.viewer.gaussian_model.transform_with_vectors(
            idx,
            scale=self.model_size_sliders[idx].value,
            r_wxyz=model_pose.wxyz,
            t_xyz=model_pose.position,
        )
        self.viewer.viewer_renderer.gaussian_model = self.viewer.gaussian_model
        self.viewer.viewer_renderer.update_pc_features()

    def _make_t_xyz_text_callback(
            self,
            idx: int,
            handle: viser.GuiInputHandle,
    ):
        @handle.on_update
        def _(event: viser.GuiEvent) -> None:
            if event.client is None:
                return

            with self.server.atomic():
                t = np.asarray(handle.value)
                if idx in self.model_transform_controls:
                    self.model_transform_controls[idx].position = t
                self.model_poses[idx].position = t

                self._transform_model(idx)
                self.viewer.rerender_for_all_client()

    def _make_r_xyz_text_callback(
            self,
            idx: int,
            handle: viser.GuiInputHandle,
    ):
        @handle.on_update
        def _(event: viser.GuiEvent) -> None:
            if event.client is None:
                return

            with self.server.atomic():
                radians = np.radians(np.asarray(handle.value))
                so3 = vst.SO3.from_rpy_radians(*radians.tolist())
                wxyz = np.asarray(so3.wxyz)
                if idx in self.model_transform_controls:
                    self.model_transform_controls[idx].wxyz = wxyz
                self.model_poses[idx].wxyz = wxyz

            self._transform_model(idx)
            self.viewer.rerender_for_all_client()

    @staticmethod
    def quaternion_to_euler_angle_vectorized2(wxyz):
        xyzw = np.zeros_like(wxyz)
        xyzw[[0, 1, 2, 3]] = wxyz[[1, 2, 3, 0]]
        euler_radians = vst.SO3.from_quaternion_xyzw(xyzw).as_rpy_radians()
        return math.degrees(euler_radians.roll), math.degrees(euler_radians.pitch), math.degrees(euler_radians.yaw)
