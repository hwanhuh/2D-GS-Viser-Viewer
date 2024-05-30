# Simple Viser Viewer for 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Original Github](https://github.com/hbb1/2d-gaussian-splatting) |<br>

This repo contains the *unofficial* viewer for the "2D Gaussian Splatting for Geometrically Accurate Radiance Fields". 

A significant portion of the viewer is based on [Original 2D GS Github](https://github.com/hbb1/2d-gaussian-splatting), [Viser](https://github.com/nerfstudio-project/viser) and [Gaussian Splatting Pytorch Lightning](https://github.com/yzslab/gaussian-splatting-lightning).

![visualization](assets/viser_capture.gif)

## Installation

- You have to follow original installation instructions in [2D GS](https://github.com/hbb1/2d-gaussian-splatting) 
- then add all the files in this project into the original project
- you should replace the original './gaussian_renderer/__init__.py' to the './gaussian_renderer/__init__.py' on this project

```bash
pip install viser
pip install splines  
pip install lightning
```

### ⭐IMPORTANT 

- currently, you have to fix line 135 of '_message_api.py' in viser (I don't know why the bug occurs)

- Original
```python
    def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
        if not isinstance(vector, tuple):
            assert cast(onp.ndarray, vector).shape == (
                length,
        ), f"Expected vector of shape {(length,)}, but got {vector.shape} instead"
    return cast(TVector, tuple(map(float, vector)))
```

- Fix it as follows 
```python 
    def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
        if isinstance(vector, tuple): return cast(TVector, vector)
        else:
            if vector.__class__.__name__ == 'RollPitchYaw':
            x = vector.roll 
            y = vector.pitch 
            z = vector.yaw
            return cast(TVector, (x, y, z))
        else:
            vector = tuple(vector)
            return cast(TVector, vector)

```



## Usage
```bash
python viewer.py <path to pre-trained model>
```
Keyboard 
- q/e for up & down
- w/a/s/d for moving
- mouse wheel for zoom in/out


## Acknowledgements
Currently, the viewer only supports just viewing
This project is built upon [Original 2D GS Github](https://github.com/hbb1/2d-gaussian-splatting), [Viser](https://github.com/nerfstudio-project/viser) and [Gaussian Splatting Pytorch Lightning](https://github.com/yzslab/gaussian-splatting-lightning).
