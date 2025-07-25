# FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis (SVD-based FloVD)<br>
<br>

![Teaser image 1](./docs/teaser.png)

[\[Project Page\]](https://jinwonjoon.github.io/flovd_site/)
[\[arXiv\]](https://arxiv.org/abs/2502.08244/)

**FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis**<br>
Wonjoon Jin, Qi Dai, Chong Luo, Seung-Hwan Baek, Sunghyun Cho<br>
POSTECH, Microsoft Research Asia
<br>

## News
* Our paper has been accepted to CVPR 2025!
* We release CogVideoX-based FloVD. Check this out! [FloVD-CogVideoX](https://github.com/JinWonjoon/FloVD/tree/cogvideox)
<br>


## Gallery

### FloVD-CogVideoX-5B
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/a55d1c29-6682-417d-886c-695b1d1b61fd" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4def8617-063f-4e61-969a-fd0507dbdeec" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/55745611-fea3-4f3f-bdd1-48b5f6c24f98" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/97be3121-ae38-45f9-822a-e387cf262824" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


## Abstract
*We present FloVD, a novel video diffusion model for camera-controllable video generation. FloVD leverages optical flow to represent the motions of the camera and moving objects. This approach offers two key benefits. Since optical flow can be directly estimated from videos, our approach allows for the use of arbitrary training videos without ground-truth camera parameters. Moreover, as background optical flow encodes 3D correlation across different viewpoints, our method enables detailed camera control by leveraging the background motion. To synthesize natural object motion while supporting detailed camera control, our framework adopts a two-stage video synthesis pipeline consisting of optical flow generation and flow-conditioned video synthesis. Extensive experiments demonstrate the superiority of our method over previous approaches in terms of accurate camera control and natural object motion synthesis.*
<br>


## TODO
- [x] Release SVD-based FloVD codes
- [x] Release evaluation benchmark dataset for object motion synthesis quality (SVD backbone)
- [x] Release CogVideoX-based FloVD codes
- [x] Release evaluation benchmark dataset for object motion synthesis quality (CogVideoX backbone)
<br>

## Preparation
* Environment (Python==3.10; CUDA==12.1; torch==2.4.1)
```shell
conda create -n flovd python=3.10.6 -y
source activate flovd
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

* Build Grounded_SAM2 (Segmentation model)
```shell
bash build_grounded_sam2.sh
```


* Checkpoints <br>
Download the FloVD checkpoints below<br>
[\[FVSM_EDM\]](https://drive.google.com/file/d/1Iw8dEGa7sd_7EHdAYMZRnlr3rxM1nmV_/view?usp=drive_link)
[\[FVSM_Quadratic\]](https://drive.google.com/file/d/1oYv3l5KIvgh6gc109BivlBaBHArWq2Sd/view?usp=drive_link)
[\[OMSM\]](https://drive.google.com/file/d/1FAKXRBK95TCf6WA6UXTKhGphoOkuf6km/view?usp=drive_link) <br>
In addition, we used the pre-trained video diffusion model (SVD), the off-the-shelf depth estimation model (Depth Anything V2, metric depth) and the segmentation model (Grounded SAM 2, open-vocabulary segmentation method).
For these models, please refer links below. <br>
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

## Tips
* Provided inference code will save depth-warped images using the input camera parameters. You can forecast the camera control results with the warped images. If the translation vector in the camera parameter is too large, you can adjust the 'speed' term in the inference code.
* For better camera controllability, you might use the FVSM-Quadratic model. For better video synthesis quality, we recommend you to use the FVSM-EDM model.
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


## Evaluation
* For the evaluation of the object motion synthesis quality, use the benchmark datasets below. 
* We provide two benchmark datasets, one for SVD and another for CogVideoX. <br>
Motion_eval_benchmark_SVD and Motion_eval_benchmark_CogVideox include video clips with 14 frames and 49 frames, respectively. For Motion_eval_benchmark_CogVideox, we use video clips with 16 fps. <br>
[\[Motion_eval_benchmark_SVD\]](https://drive.google.com/file/d/1kdcJFqdCsg5OlBK4VCjqvXA-ezb0gANw/view?usp=drive_link)
[\[Motion_eval_benchmark_CogVideoX\]](https://drive.google.com/file/d/1EKNtH72reT3MxRHHB83PVaxaDYeX6VL1/view?usp=drive_link)
* For detailed description about the evaluation protocol, please refer to the Sec. 5.2 of the main paper.
* If you use the benchmark datasets of the object motion synthesis quality, please cite our paper. <br>


## Others
* We heavily borrow codes from [\[CameraCtrl\]](https://github.com/hehao13/CameraCtrl). Thanks for their contributions.

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install python-dateutil
export TORCH_CUDA_ARCH_LIST="9.0"
pip install ninja -> pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
sudo cp nvidia_video_sdk/Lib/linux/stubs/aarch64/libnvcuvid.so /usr/bin/nvcc OR /usr/lib/aarch64-linux-gnu/ 
rm -rf *
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make
cd python -> pip install -e . --no-deps


```bibtex
@article{jin2025flovd,
         title={FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis},
         author={Jin, Wonjoon and Dai, Qi and Luo, Chong and Baek, Seung-Hwan and Cho, Sunghyun},
         journal={arXiv preprint arXiv:2502.08244},
         year={2025}
}
```
