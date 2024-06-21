# Simple Viser Viewer for 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[2D GS Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Original Github](https://github.com/hbb1/2d-gaussian-splatting) <br>

This repo contains the *unofficial* viewer for the "2D Gaussian Splatting for Geometrically Accurate Radiance Fields". 
A significant portion of this project is built upon several existing works to provide enhanced visualization and editing capabilities for 2D Gaussian Splatting data.
![visualization](assets/viser_teaser.gif)

## ⭐ Features  
|  Rendering  | Training |
| --- | --- |
| <img src="assets/viser_train.gif" width="450"/> | <img src="assets/viser_train2.gif" width="450"/> |

| General | Edit  | Transform |
| --- | --- | --- |
| <img src="assets/viser_general_opt.gif" width="300"/> | <img src="assets/viser_edit_opt.gif" width="300"/> | <img src="assets/viser_transform_opt.gif" width="300"/> |

- Various Render Type: RGB / Edge / Normal / View-Normal / Depth / Depth-to-Normal / Depth-Distortion / Curvature
- Edit & Save Splats
- Mesh Export
- Render Path and Preview  

## Updates History
- 2024/06/21
    - Mesh Export 
- 2024/06/17
    - Improve fps
    - Minor code revision
- 2024/06/07
    - Render Types Update
        - Now supports ***Edge/Curvature/Depth-Distortion*** render type
            - The edge/curvature visualization is inspired by [Gaussian Splatting Monitor](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor). 
        - For one channel output image (*e.g.,* depth), use the 'Turbo' color map for the better visualization 
        - **Note**. Depth-distortion visualization is quite interesting. In the early stages of training, it shows noisy and misaligned splats, but after training, splats align along the depth, highlighting 'edges' on the view frustum.
    - Bug Fixes 
        - Various render options are available during training 
        - Transform after editing
- 2024/06/05
    - Training / Rendering Features
        - Now supports training with a viewer (large memory consumption)
        - Supports render path generation & preview camera paths 
    - Minor code revision
        - Add 'Set to default' in the transform panel 
        - Cropbox w/ multi slider
        - Edit mode visualizes only pointcloud for clarity
        - Fix negative parts truncation in normal rendering
- 2024/06/03
    - General Features
        - Supports various render types including ***Normal / Depth / Depth2Normal***
        - Direct comparison between different render types, *e.g.,* normal vs depth-to-normal
        - Cropbox Region
        - Pointcloud visualization
    - Editing Features
        - Edit, delete and save point clouds ~~(Recommended with: 'Add pointcloud')~~
    - Transform Features
        - Rigid Transform 
- 2024/05/31
    - Viewer release

## Installation

- Follow the original installation instructions from 2D GS, then add all files from this project to the original project directory. 
(*i.e.,* 'viewer.py' is in '~/2d-gaussian-splatting/viewer.py')
- Tested on Ubuntu 20.04.

```bash
pip install viser==0.1.29
pip install splines  
pip install lightning
```
- **Note** sorry for the confusion, the previous instruction for the '_message_api.py' can be resolved with viser==0.1.29 

## Usage
- View a 2D GS ply file 
```bash
python viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
### enable transform mode
python viewer.py <path to pre-trained model> -s <data source path> --enable_transform
```
- Train w/ viewer
```bash
python train_w_viewer.py -s <path to datas>
```
- Colab
    - You can also use the viewer in the Colab, powered by ngrok (see [example](./2dgs_viewer_colab.ipynb))
    - To use Colab and ngrok, you should add the below code to the 'start' function in the 'viewer.py' (line 246)
```python
    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        self._setup_titles(server)
        if server_config_fun is not None:
            server_config_fun(self, server)

        ### attach here!!!
        from pyngrok import ngrok
        authtoken = "your authtoken"
        ngrok.set_auth_token(authtoken)
        public_url = ngrok.connect(self.port)
        print(f"ngrok tunnel URL: {public_url}")
        ### 
```

### Control 
- **'q/e'** for up & down
- **'w/a/s/d'** for moving
- Mouse wheel for zoom in/out


## Acknowledgements
This project is built upon the following works
- [Original 2D GS Github](https://github.com/hbb1/2d-gaussian-splatting)
- [Viser](https://github.com/nerfstudio-project/viser)
- [Gaussian Splatting Pytorch Lightning](https://github.com/yzslab/gaussian-splatting-lightning).

Realted Blog Post: [Review of 2D Gaussian Splatting (Korean)](https://velog.io/@gjghks950/Review-2D-Gaussian-Splatting-for-Geometrically-Accurate-Radiance-Fields-Viewer-%EA%B5%AC%ED%98%84-%EC%86%8C%EA%B0%9C)
