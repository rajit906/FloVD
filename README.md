# FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis<br>
<sub> SVD-based FloVD codes </sub>
<br>

![Teaser image 1](./docs/teaser.png)

[\[Project Page\]](https://jinwonjoon.github.io/flovd_site/)
[\[arXiv\]](https://arxiv.org/abs/2502.08244/)

**FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis**<br>
Wonjoon Jin, Qi Dai, Chong Luo, Seung-Hwan Baek, Sunghyun Cho<br>
POSTECH, Microsoft Research Asia
<br>

## Abstract
*We present FloVD, a novel video diffusion model for camera-controllable video generation. FloVD leverages optical flow to represent the motions of the camera and moving objects. This approach offers two key benefits. Since optical flow can be directly estimated from videos, our approach allows for the use of arbitrary training videos without ground-truth camera parameters. Moreover, as background optical flow encodes 3D correlation across different viewpoints, our method enables detailed camera control by leveraging the background motion. To synthesize natural object motion while supporting detailed camera control, our framework adopts a two-stage video synthesis pipeline consisting of optical flow generation and flow-conditioned video synthesis. Extensive experiments demonstrate the superiority of our method over previous approaches in terms of accurate camera control and natural object motion synthesis.*
<br>

## News
* Our paper has been accepted to CVPR 2025!
<br>

## TODO
- [x] Release SVD-based FloVD codes
- [ ] Release evaluation benchmark dataset for object motion synthesis quality (SVD backbone)
- [ ] Release CogVideoX-based FloVD codes
- [ ] Release evaluation benchmark dataset for object motion synthesis quality (CogVideoX backbone)
<br>

## Preparation
* Environment (Python==3.10; CUDA==12.1; torch==2.4.1)
```shell
conda create -n flovd python=3.10.6 -y
source activate flovd
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

* Build Grounded_SAM2 (Segmentation module)
```shell
bash build_grounded_sam2.sh
```


* Checkpoints <br>
Download the checkpoints below<br>
[\[FVSM_EDM\]](https://drive.google.com/file/d/1Iw8dEGa7sd_7EHdAYMZRnlr3rxM1nmV_/view?usp=drive_link)
[\[FVSM_Quadratic\]](https://drive.google.com/file/d/1oYv3l5KIvgh6gc109BivlBaBHArWq2Sd/view?usp=drive_link)
[\[OMSM\]](https://drive.google.com/file/d/1FAKXRBK95TCf6WA6UXTKhGphoOkuf6km/view?usp=drive_link) <br>
In addition, we used pre-trained video diffusion model (SVD), off-the-shelf depth estimation model (Depth Anything V2, metric depth) and segmentation model (Grounded SAM 2, open-vocabulary segmentation method).
For these models, please refer links below.
[\[SVD\]](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/tree/main)
[\[Depth_anything_v2_metric\]](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
[\[Grounded_SAM2\]](https://github.com/IDEA-Research/Grounded-SAM-2)
<br>


## Inference
* Preparation <br>
Before sampling, set path (configuration, checkpoint) in the bash script
Before sampling, set video data for inference. You need only one frame per scene for the input image.
```shell
# File tree
./[data_root]/
├── frames
│   ├── [scene_name]
│   ├   ├── 00.png
│   ├   ├── ... (not_necessary)
```

* Sample video frames <br>
FloVD synthesizes 14-frame videos.
```shell
bash scripts/inference_FloVD.sh
```
<br>

* Tips <br>
Provided inference code will save depth-warped images using the input camera parameters. 
You can forecast the camera control results with the warped images.
If the translation vector in the camera parameter is too large, you can adjust the 'speed' term in the inference code.
<br>


## Training FloVD
* Training Dataset
```shell
# Prepare your own dataset
# File tree
# metadata.json includes path list to each video data
./[data_root]/
├── metadata
│   ├ metadata.json
├── video
│   ├── xxxxx.mp4
│   ├── ...
```

* Preparation <br>
Before training, set path (SVD backbone, Dataset, Depth_anything_v2, Grounded_SAM2) in the configuration files.

* FVSM
```shell
bash scripts/train_FVSM.sh
```

* OMSM
```shell
bash scripts/train_OMSM.sh
bash scripts/train_OMSM_Curated.sh
```

## Others
* We heavily borrow codes from [\[CameraCtrl\]](https://github.com/hehao13/CameraCtrl). Thanks for their contributions.


```bibtex
@article{jin2025flovd,
         title={FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis},
         author={Jin, Wonjoon and Dai, Qi and Luo, Chong and Baek, Seung-Hwan and Cho, Sunghyun},
         journal={arXiv preprint arXiv:2502.08244},
         year={2025}
}
```