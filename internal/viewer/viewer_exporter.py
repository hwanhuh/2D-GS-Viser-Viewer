import os
import sys 
import torch
import open3d as o3d
import json
import logging 
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from argparse import ArgumentParser, Namespace

from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from arguments import ModelParams, PipelineParams

def custom_combined_args(parser : ArgumentParser):
    if not sys.argv[0] == 'viewer.py':
        cmdlne_string = sys.argv[1:]
        cfgfile_string = "Namespace()"
        args_cmdline = parser.parse_args(cmdlne_string)
        try:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
        except TypeError:
            print("Config file not found at")
            pass
    else:
        args_cmdline = parser
        cfgfile_string = f"{(args_cmdline)}"
    
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class MeshExporter:
    def __init__(self, 
                args, 
                gaussians,
                iteration,
                model_params,
                pipeline_params):
        self.args = args
        self.model_params = model_params 
        self.pipeline_params = pipeline_params

        self.dataset, self.pipe = self.load_data()
        self.iteration = iteration
        self.gaussians = gaussians
        self.bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.gaussExtractor = GaussianExtractor(self.gaussians, render, self.pipe, bg_color=self.bg_color)
        self.model_path = args.model_path
        self.load_train_cameras(self.dataset)

    def load_data(self):
        dataset = self.model_params.extract(self.args)
        pipe = self.pipeline_params.extract(self.args)
        return dataset, pipe

    def load_train_cameras(self, dataset_args):
        if os.path.exists(os.path.join(dataset_args.source_path, "sparse")):
            print("Found COLMAP datas, Loading Datasets")
            scene_info = sceneLoadTypeCallbacks["Colmap"](dataset_args.source_path, dataset_args.images, dataset_args.eval)
        elif os.path.exists(os.path.join(dataset_args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](dataset_args.source_path, dataset_args.white_background, dataset_args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1., dataset_args)

    def start_logging(self, log_filename='logfile.txt'):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_filename)
            ]
        )
        self.logger = logging.getLogger(__name__)
        sys.stdout = StreamToLogger(self.logger, logging.INFO)

    def stop_logging(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def export_mesh(self):
        train_dir = os.path.join(self.model_path, 'train', f"ours_{self.iteration}")
        os.makedirs(train_dir, exist_ok=True)
        
        self.gaussExtractor.gaussians.active_sh_degree = 0
        self.gaussExtractor.reconstruction(self.train_cameras)
        
        if self.args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = self.gaussExtractor.extract_mesh_unbounded(resolution=self.args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (self.gaussExtractor.radius * 2.0) if self.args.depth_trunc < 0 else self.args.depth_trunc
            voxel_size = (depth_trunc / self.args.mesh_res) if self.args.voxel_size < 0 else self.args.voxel_size
            sdf_trunc = 5.0 * voxel_size if self.args.sdf_trunc < 0 else self.args.sdf_trunc
            mesh = self.gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print(f"mesh saved at {os.path.join(train_dir, name)}")
        mesh_post = post_process_mesh(mesh, cluster_to_keep=self.args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        final_mesh_path = os.path.join(train_dir, name.replace('.ply', '_post.ply'))
        print(f"mesh post processed saved at {final_mesh_path}")
        return final_mesh_path

    @staticmethod
    def parse_args_mesh(model_path: str, 
                        source_path: str, 
                        args, 
                        unbounded: bool=False, 
                        mesh_res: int=1024, 
                        num_cluster: int=50, 
                        ):
        parser_mesh = ArgumentParser(description="Mesh Export Parameters")
        model_params = ModelParams(parser_mesh)
        pipeline_params = PipelineParams(parser_mesh)
        parser_mesh.set_defaults(model_path=model_path, source_path=source_path, unbounded=unbounded, mesh_res=mesh_res, num_cluster=num_cluster)
        parser_mesh.add_argument("--iteration", default=-1, type=int)
        parser_mesh.add_argument("--quiet", action="store_true")
        parser_mesh.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
        parser_mesh.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
        parser_mesh.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
        # parser_mesh.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
        # parser_mesh.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
        # parser_mesh.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')

        # Convert args Namespace object to a list of strings
        arg_strings = []
        for key, value in vars(args).items():
            arg_strings.append(f"--{key}")
            arg_strings.append(str(value))

        # Parse arguments
        parsed_args, _ = parser_mesh.parse_known_args(arg_strings)
        return custom_combined_args(parsed_args), model_params, pipeline_params
