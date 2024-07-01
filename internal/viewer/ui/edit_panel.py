import traceback
import datetime
import os.path

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re
import threading
import time
import trimesh
from internal.viewer import MeshExporter


class EditPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab
        self.C0 = 0.28209479177387814
        self.mesh = None
        self.mesh_path = None

        self._setup_point_cloud_folder()
        self._setup_gaussian_edit_folder()
        self._setup_save_gaussian_folder()
        self._setup_mesh_export_folder()
        self.export_mesh_block()

    def _setup_point_cloud_folder(self):
        server = self.server
        with self.server.add_gui_folder("Point Cloud"):
            self.show_point_cloud_checkbox = server.add_gui_checkbox(
                "Edit (w/ ptc)",
                initial_value=False,
            )
            self.point_size = server.add_gui_number(
                "Point Size",
                min=0.001,
                initial_value=0.005,
            )
            self.point_sparsify = server.add_gui_number(
                "Point Sparsity",
                min=1,
                initial_value=3,
            )
            self.pcd = None

            @self.show_point_cloud_checkbox.on_update
            @self.point_size.on_update
            @self.point_sparsify.on_update
            def _(event: viser.GuiEvent):
                with self.server.atomic():
                    self._update_pcd()

    def _resize_grid(self, idx):
        exist_grid = self.grids[idx][0]
        exist_grid.remove()
        self.grids[idx][0] = self.server.add_grid(
            "/grid/{}".format(idx),
            width=self.grids[idx][2].value[0],
            height=self.grids[idx][2].value[1],
            wxyz=self.grids[idx][1].wxyz,
            position=self.grids[idx][1].position,
        )
        self._update_scene()

    def _setup_gaussian_edit_folder(self):
        server = self.server

        self.edit_histories = []

        with server.add_gui_folder("Edit"):
            # initialize a list to store panel(grid)'s information
            self.grids: dict[int, list[
                viser.MeshHandle,
                viser.TransformControlsHandle,
                viser.GuiInputHandle,
            ]] = {}
            self.grid_idx = 0

            add_grid_button = server.add_gui_button("Add Panel")
            self.delete_gaussians_button = server.add_gui_button(
                "Delete Gaussians",
                color="red",
            )

        self.grid_folders = {}

        # create panel(grid)
        def new_grid(idx):
            with self.server.add_gui_folder("Grid {}".format(idx)) as folder:
                self.grid_folders[idx] = folder

                # TODO: add height
                grid_size = server.add_gui_vector2("Size", initial_value=(10., 10.), min=(0., 0.), step=0.01)

                grid = server.add_grid(
                    "/grid/{}".format(idx),
                    height=grid_size.value[0],
                    width=grid_size.value[1],
                )
                grid_transform = server.add_transform_controls(
                    "/grid_transform_control/{}".format(idx),
                    wxyz=grid.wxyz,
                    position=grid.position,
                )

                # resize panel on size value changed
                @grid_size.on_update
                def _(event: viser.GuiEvent):
                    with event.client.atomic():
                        self._resize_grid(idx)

                # handle panel deletion
                grid_delete_button = server.add_gui_button("Delete")

                @grid_delete_button.on_click
                def _(_):
                    with server.atomic():
                        try:
                            self.grids[idx][0].remove()
                            self.grids[idx][1].remove()
                            self.grids[idx][2].remove()
                            self.grid_folders[idx].remove()  # bug
                        except Exception as e:
                            traceback.print_exc()
                        finally:
                            del self.grids[idx]
                            del self.grid_folders[idx]

                    self._update_scene()

            # update the pose of panel(grid) when grid_transform updated
            @grid_transform.on_update
            def _(_):
                self.grids[idx][0].wxyz = grid_transform.wxyz
                self.grids[idx][0].position = grid_transform.position
                self._update_scene()

            self.grids[self.grid_idx] = [grid, grid_transform, grid_size]
            self._update_scene()

        # setup callbacks

        @add_grid_button.on_click
        def _(_):
            with server.atomic():
                new_grid(self.grid_idx)
                self.grid_idx += 1

        @self.delete_gaussians_button.on_click
        def _(_):
            with server.atomic():
                gaussian_to_be_deleted, pose_and_size_list = self._get_selected_gaussians_mask(return_pose_and_size_list=True)
                self.edit_histories.append(pose_and_size_list)
                self.viewer.gaussian_model.delete_gaussians(gaussian_to_be_deleted)
                self._update_pcd()
            self.viewer.viewer_renderer.gaussian_model = self.viewer.gaussian_model
            self.viewer.viewer_renderer.update_pc_features()
            self.viewer.rerender_for_all_client()

    def _setup_save_gaussian_folder(self):
        with self.server.add_gui_folder("Save"):
            name_text = self.server.add_gui_text(
                "Name",
                initial_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            save_button = self.server.add_gui_button("Save")

            @save_button.on_click
            def _(event: viser.GuiEvent):
                # skip if not triggered by client
                if event.client is None:
                    return
                try:
                    save_button.disabled = True

                    with self.server.atomic():
                        try:
                            # check whether is a valid name
                            name = name_text.value
                            match = re.search("^[a-zA-Z0-9_\-]+$", name)
                            if match:
                                output_directory = "edited"
                                os.makedirs(output_directory, exist_ok=True)
                                try:
                                    if len(self.edit_histories) > 0:
                                        torch.save(self.edit_histories, os.path.join(output_directory, f"{name}-edit_histories.ckpt"))
                                except:
                                    traceback.print_exc()

                                # save ply
                                ply_save_path = os.path.join(output_directory, "{}.ply".format(name))
                                self.viewer.gaussian_model.save_ply(ply_save_path)
                                message_text = "Saved to {}".format(ply_save_path)
                            else:
                                message_text = "Invalid name"
                        except:
                            traceback.print_exc()

                    # show message
                    with event.client.add_gui_modal("Message") as modal:
                        event.client.add_gui_markdown(message_text)
                        close_button = event.client.add_gui_button("Close")

                        @close_button.on_click
                        def _(_) -> None:
                            modal.close()

                finally:
                    save_button.disabled = False

    def _get_selected_gaussians_mask(self, return_pose_and_size_list: bool = False):
        xyz = self.viewer.gaussian_model.get_xyz

        # if no grid exists, do not delete any gaussians
        if len(self.grids) == 0:
            return torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.bool)

        pose_and_size_list = []
        # initialize mask with True
        is_gaussian_selected = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.bool)
        for i in self.grids:
            # get the pose of grid, and build world-to-grid transform matrix
            grid = self.grids[i][0]
            se3 = torch.linalg.inv(torch.tensor(vtf.SE3.from_rotation_and_translation(
                vtf.SO3(grid.wxyz),
                grid.position,
            ).as_matrix()).to(xyz))
            # transform xyz from world to grid
            new_xyz = torch.matmul(xyz, se3[:3, :3].T) + se3[:3, 3]
            # find the gaussians to be deleted based on the new_xyz
            grid_size = self.grids[i][2].value
            x_mask = torch.abs(new_xyz[:, 0]) < grid_size[0] / 2
            y_mask = torch.abs(new_xyz[:, 1]) < grid_size[1] / 2
            z_mask = new_xyz[:, 2] > 0
            # update mask
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, x_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, y_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, z_mask)

            # add to history
            pose_and_size_list.append((se3.cpu(), grid_size))

        if return_pose_and_size_list is True:
            return is_gaussian_selected, pose_and_size_list
        return is_gaussian_selected

    def _get_selected_gaussians_indices(self):
        """
        get the index of the gaussians which in the range of grids
        :return:
        """
        selected_gaussian = torch.where(self._get_selected_gaussians_mask())
        return selected_gaussian

    def _update_pcd(self, selected_gaussians_indices=None):
        self.remove_point_cloud()
        if self.show_point_cloud_checkbox.value is False:
            return
        xyz = self.viewer.gaussian_model.get_xyz

        # get SH0 colors
        dc = self.viewer.gaussian_model._features_dc.clone() 
        dc = (self.C0 * dc + 0.5).clip(min=0.0, max=1.0)
        dc = (255 * dc[:, 0]).to(torch.uint8)
        colors = dc
        if selected_gaussians_indices is None:
            selected_gaussians_indices = self._get_selected_gaussians_indices()
        colors[selected_gaussians_indices] = 255 - colors[selected_gaussians_indices]

        point_sparsify = int(self.point_sparsify.value)
        self.show_point_cloud(xyz[::point_sparsify].cpu().detach().numpy(), colors[::point_sparsify].cpu().detach().numpy())

    def remove_point_cloud(self):
        if self.pcd is not None:
            self.pcd.remove()
            self.pcd = None

    def show_point_cloud(self, xyz, colors):
        self.pcd = self.server.add_point_cloud(
            "/pcd",
            points=xyz,
            colors=colors,
            point_size=self.point_size.value,
        )

    def _update_scene(self):
        selected_gaussians_indices = self._get_selected_gaussians_mask()
        self.viewer.gaussian_model.select(selected_gaussians_indices)
        self._update_pcd(selected_gaussians_indices)

        self.viewer.rerender_for_all_client()

    def _setup_mesh_export_folder(self):
        with self.server.add_gui_folder("Mesh Export"):
            self.unbounded = self.server.add_gui_checkbox(
                "unbounded",
                initial_value=False,
            )
            self.num_cluster = self.server.add_gui_slider(
                "Num Cluster",
                min=10,
                max=200,
                step=10,
                initial_value=50,
            )
            self.mesh_res = self.server.add_gui_slider(
                "Mesh Res",
                min=128,
                max=2048,
                step=128,
                initial_value=1024,
            )
            self.export_button = self.server.add_gui_button("Export Mesh", color="green", icon=viser.Icon.FILE_EXPORT)
            self.show_mesh_button = self.server.add_gui_button("Show Mesh Result", color="yellow", icon=viser.Icon.PLAYER_PLAY)
            self.unshow_mesh_button = self.server.add_gui_button("unshow Mesh", color="yellow", icon=viser.Icon.PLAYER_PLAY, visible=False)

    def export_mesh_block(self):
        self._default_export_log = f"**Model path**: {self.viewer.model_paths} \\\n  **Data path**: {self.viewer.source_path})"
        self._mesh_export_log_dir = os.path.join(os.getcwd(), 'temp/mesh_log.txt')
        if os.path.exists(self._mesh_export_log_dir):
            os.remove(self._mesh_export_log_dir)
        dir_path = os.path.dirname(self._mesh_export_log_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Mesh Export !!! 
        def read_last_lines(file_path, num_lines):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                formatted_lines = [' \\\n [Info]' + line.rstrip() for line in lines[-num_lines:-1]]
                return ''.join(formatted_lines)

        @self.export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            with self.server.atomic():
                with event.client.add_gui_modal("[Mesh Export]") as modal:
                    self.export_mesh_text = event.client.add_gui_markdown(self._default_export_log)
                    close_button = event.client.add_gui_button("Close", visible=False)
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()

            # Update the GUI first before starting the thread
            def update_gui_and_start_export():
                def export() -> None:
                    with open(self._mesh_export_log_dir, 'w') as file:
                        file.write("mesh exporting... \\\n ")
                    ex_args, model_params, export_pipe_params = MeshExporter.parse_args_mesh(self.viewer.model_paths, 
                                                                                            self.viewer.source_path, 
                                                                                            self.viewer.args, 
                                                                                            unbounded = self.unbounded.value, 
                                                                                            mesh_res = self.mesh_res.value, 
                                                                                            num_cluster = self.num_cluster.value)
                    mesh_exporter = MeshExporter(ex_args, self.viewer.gaussian_model, self.viewer.iteration, model_params, export_pipe_params)
                    mesh_exporter.start_logging(self._mesh_export_log_dir)
                    self.mesh_path = mesh_exporter.export_mesh()
                    mesh_exporter.stop_logging()

                def update_log() -> None:
                    while export_thread.is_alive():
                        self.export_mesh_text.content = self._default_export_log + read_last_lines(self._mesh_export_log_dir, 15)
                        time.sleep(0.1)
                    
                export_thread = threading.Thread(target=export)
                log_thread = threading.Thread(target=update_log)
                export_thread.start()
                log_thread.start()
                export_thread.join()
                log_thread.join()
                self.export_mesh_text.content = f'Done! \n Your Mesh is saved at: {self.mesh_path}'
                close_button.visible = True

            # Call the function to update the GUI and start the export process
            update_gui_and_start_export()

        @self.show_mesh_button.on_click
        def _(event: viser.GuiEvent) -> None:
            with self.server.atomic():
                self.show_mesh_button.visible = False 
                self.unshow_mesh_button.visible = True

                if self.mesh is None and self.mesh_path is not None:
                    mesh = trimesh.load_mesh(self.mesh_path)
                    mesh.apply_transform(np.linalg.inv(self.viewer.camera_transform.cpu().numpy()))
                    self.mesh = self.server.add_mesh_trimesh(
                        name="/trimesh",
                        mesh=mesh, 
                        scale = 1,
                        wxyz=event.client.camera.wxyz,
                        position=tuple(self.viewer.camera_center),
                    )
                    self.mesh_control = event.client.add_transform_controls(
                        f"/mesh_control",
                        scale=0.5,
                        wxyz=self.mesh.wxyz,
                        position=self.mesh.position,
                    )
                    def _make_mesh_controls_callback(
                            trimesh_obj,
                            control: viser.TransformControlsHandle,
                    ) -> None:
                        @control.on_update
                        def _(_) -> None:
                            trimesh_obj.wxyz = control.wxyz
                            trimesh_obj.position = control.position
                            print(control.wxyz, control.position)
                    _make_mesh_controls_callback(self.mesh, self.mesh_control)
                else:
                    with event.client.add_gui_modal("[Alert]") as modal:
                        self.export_mesh_text = event.client.add_gui_markdown("<sub> There is no exported mesh </sub>")
                        close_button = event.client.add_gui_button("Close", visible=True)
                        @close_button.on_click
                        def _(_) -> None:
                            modal.close()

        @self.unshow_mesh_button.on_click
        def _(event: viser.GuiEvent) -> None:
            with self.server.atomic():
                self.show_mesh_button.visible = True 
                self.unshow_mesh_button.visible = False
                if self.mesh is not None:
                    self.mesh.remove()
                    self.mesh_control.remove()
                    self.mesh = None
