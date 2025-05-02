from typing import Any, Dict, List, Tuple
from pathlib import Path
import os
import hashlib
import json
import random
import wandb
import math
import numpy as np
from einops import rearrange, repeat
from safetensors.torch import load_file, save_file
from accelerate.logging import get_logger

import torch

from accelerate.utils import gather_object

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils.export_utils import export_to_video

from finetune.pipeline.flovd_OMSM_cogvideox_pipeline import FloVDOMSMCogVideoXImageToVideoPipeline
from finetune.constants import LOG_LEVEL, LOG_NAME

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Args, Components, State
from finetune.trainer import Trainer
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_memory_statistics,
    string_to_filename,
    unwrap_model,
)
from finetune.datasets.utils import (
    preprocess_image_with_resize,
    load_binary_mask_compressed,
)
from finetune.modules.camera_sampler import SampleManualCam
from finetune.modules.camera_flow_generator import CameraFlowGenerator
from finetune.modules.utils import get_camera_flow_generator_input, forward_bilinear_splatting, flow_to_color

from ..utils import register

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

logger = get_logger(LOG_NAME, LOG_LEVEL)

class FloVDOMSMCogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def __init__(self, args: Args) -> None:
        super().__init__(args)


    @override
    def load_components(self) -> Dict[str, Any]:
        # TODO. Change the pipeline and ...
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = FloVDOMSMCogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components
    

    @override
    def initialize_pipeline(self) -> FloVDOMSMCogVideoXImageToVideoPipeline:
        # TODO. Change the pipeline and ...
        pipe = FloVDOMSMCogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    def initialize_flow_generator(self):
        depth_estimator_kwargs = {
            "target": 'modules.depth_warping.depth_warping.DepthWarping_wrapper',
            "kwargs": {
                "ckpt_path": '/workspace/workspace/checkpoints/depth_anything/depth_anything_v2_metric_hypersim_vitb.pth',
                "model_config": {
                    "max_depth": 20,
                    "encoder": 'vitb',
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                }

            }
        }

        return CameraFlowGenerator(depth_estimator_kwargs)

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "encoded_flow": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            encoded_flow = sample["encoded_flow"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["encoded_flow"].append(encoded_flow)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["encoded_flow"] = torch.stack(ret["encoded_flow"])

        return ret
    

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        images = batch["images"]
        latent_flow = batch["encoded_flow"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of images: [B, C, H, W]
        # Shape of latent_flow: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t # WJ: None in i2v setting...
        if patch_size_t is not None:
            # ncopy = latent.shape[2] % patch_size_t
            # # Copy the first frame ncopy times to match patch_size_t
            # first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            # latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            # assert latent.shape[2] % patch_size_t == 0
            raise NotImplementedError("Do not use the case whose patch_size_t is not None")
        
        batch_size, num_channels, num_frames, height, width = latent_flow.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent_flow.dtype)
        
        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent_flow = latent_flow.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (image_latents.shape[0], *image_latents.shape[2:]) == (latent_flow.shape[0], *latent_flow.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent_flow.shape[0], latent_flow.shape[1] - 1, *latent_flow.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent_flow)
        latent_flow_noisy = self.components.scheduler.add_noise(latent_flow, noise, timesteps)


        # Concatenate latent and image_latents in the channel dimension
        latent_flow_img_noisy = torch.cat([latent_flow_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent_flow.new_full((1,), fill_value=2.0)
        )

        predicted_noise = self.components.transformer(
            hidden_states=latent_flow_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_flow_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent_flow) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss
    
    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

    # Validation

    @override
    def prepare_for_validation(self):
        # Load from dataset?
        # Data_root
        # - metadata.jsonl
        # - video_latent / args.resolution /
        # - prompt_embeddings /
        # - first_frames /
        # - flow_direct_f_latent /

        data_root = self.args.data_root
        metadata_path = data_root / "metadata_revised.jsonl"
        assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl or metadata_revised.jsonl in the root path"
        
        # Load metadata
        # metadata = {
        #     "video_path": ...,
        #     "hash_code": ...,
        #     "prompt": ...,
        # }
        metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                metadata.append( json.loads(line) )

        metadata = random.sample(metadata, self.args.max_scene)

        prompts = [x["prompt"] for x in metadata]
        if 'curated' in str(data_root).lower():
            self.prompt_embeddings = [data_root / "prompt_embeddings" / (x["hash_code"] + '.safetensors') for x in metadata]
        else:
            self.prompt_embeddings = [data_root / "prompt_embeddings_revised" / (x["hash_code"] + '.safetensors') for x in metadata]
        videos = [data_root / "video_latent" / "x".join(str(x) for x in self.args.train_resolution) / (x["hash_code"] + '.safetensors') for x in metadata]
        images = [data_root / "first_frames" / (x["hash_code"] + '.png') for x in metadata]
        flows = [data_root / "flow_direct_f_latent" / (x["hash_code"] + '.safetensors') for x in metadata]

        # load prompt embedding
        validation_prompts = []
        validation_prompt_embeddings = []
        validation_video_latents = []
        validation_images = []
        validation_flow_latents = []
        for prompt, prompt_embedding, video_latent, image, flow_latent in zip(prompts, prompt_embeddings, videos, images, flows):
            validation_prompts.append(prompt)
            validation_prompt_embeddings.append(load_file(prompt_embedding)["prompt_embedding"].unsqueeze(0))
            validation_video_latents.append(load_file(video_latent)["encoded_video"].unsqueeze(0))
            validation_flow_latents.append(load_file(flow_latent)["encoded_flow_f"].unsqueeze(0))
            # validation_images.append(preprocess_image_with_resize(image, self.args.train_resolution[1], self.args.train_resolution[2]))
            validation_images.append(image)
            
            
        validation_videos = [None] * len(validation_prompts)
        

        self.state.validation_prompts = validation_prompts
        self.state.validation_prompt_embeddings = validation_prompt_embeddings
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos
        self.state.validation_video_latents = validation_video_latents
        self.state.validation_flow_latents = validation_flow_latents

        # Debug..
        self.validate(0)
    
    
    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: FloVDOMSMCogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """

        prompt_embedding, image = eval_data["prompt_embedding"], eval_data["image"]

        flow_latent_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=None,
            prompt_embeds=prompt_embedding,
            image=image,
            generator=self.state.generator,
            num_inference_steps=50,
            output_type='latent'
        ).frames[0]

        flow_generate = decode_flow(flow_latent_generate.unsqueeze(0).to(self.accelerator.device), self.components.vae, flow_scale_factor=[60, 36]) # BF,C,H,W
        
        return [("synthesized_flow", flow_generate)]


    @override
    def validate(self, step: int) -> None:
        #TODO. Fix the codes!!!!
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()
        camera_flow_generator = self.initialize_flow_generator().to(device=self.accelerator.device, dtype=self.state.weight_dtype)

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################
        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]
            video_latent = self.state.validation_video_latents[i].permute(0,2,1,3,4) # [B,F,C,H,W] (e.g., [B, 13, 16, 60, 90])
            prompt_embedding = self.state.validation_prompt_embeddings[i]
            flow_latent = self.state.validation_flow_latents[i].permute(0,2,1,3,4) # [B,F,C,H,W] (e.g., [B, 13, 16, 60, 90])
            

            if image is not None:
                image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                image_torch = image.detach().clone()
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
            else:
                with torch.cuda.amp.autocast(enabled=True, dtype=self.state.weight_dtype):
                    try:
                        video_decoded = decode_latents(video_latent.to(self.accelerator.device), self.components.vae)
                    except:
                        pass
                    video_decoded = decode_latents(video_latent.to(self.accelerator.device), self.components.vae)
                video = ((video_decoded + 1.) / 2. * 255.)[0].permute(1,0,2,3).float().clip(0., 255.).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
                
                with torch.cuda.amp.autocast(enabled=True, dtype=self.state.weight_dtype):
                    try:
                        flow_decoded = decode_flow(flow_latent.to(self.accelerator.device), self.components.vae, flow_scale_factor=[60, 36])
                    except:
                        pass
                    flow_decoded = decode_flow(flow_latent.to(self.accelerator.device), self.components.vae, flow_scale_factor=[60, 36]) # (BF)CHW (C=2)


            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            # validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe)
            validation_artifacts = self.validation_step({"prompt_embedding": prompt_embedding, "image": image}, pipe)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
                
                # Log flow
                artifacts.update({f"artifact_flow_{i}": {"type": 'flow', "value": flow_decoded}})
                
                # Log flow_warped_frames
                image_tensor = repeat(rearrange(torch.tensor(np.array(image)).to(flow_decoded.device, torch.float), 'h w c -> 1 c h w'), 'b c h w -> (b f) c h w', f=flow_decoded.size(0)) # scale~(0,255) (BF) C H W
                warped_video = forward_bilinear_splatting(image_tensor, flow_decoded.to(torch.float)) # if we have an occlusion mask from dataset, we can use it.
                frame_list = []
                for frame in warped_video:
                    frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
                    frame_list.append(Image.fromarray(frame))
                
                artifacts.update({f"artifact_warped_video_{i}": {"type": 'warped_video', "value": frame_list}})
                
                # Log synthesized_flow_wraped_frames
                # artifact_value: synthesized optical flow
                warped_video2 = forward_bilinear_splatting(image_tensor, artifact_value.to(torch.float)) # if we have an occlusion mask from dataset, we can use it. For OMSM, do not use.
                frame_list2 = []
                for frame in warped_video2:
                    frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
                    frame_list2.append(Image.fromarray(frame))
                
                artifacts.update({f"artifact_synthesized_flow_warped_video_{i}": {"type": 'synthesized_flow_warped_video', "value": frame_list2}})
                
                    
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video", "flow", "warped_video", "synthesized_flow", "synthesized_flow_warped_video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                if artifact_type == "warped_video" or artifact_type == "synthesized_flow_warped_video":
                    filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}_{artifact_type}.{extension}"
                elif artifact_type == "synthesized_flow":
                    filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}_synthesized_flow.{extension}"
                elif artifact_type == "flow":
                    filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}_original_flow.{extension}"
                else:
                    filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video" or artifact_type == "warped_video" or artifact_type == "synthesized_flow_warped_video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=f"[{artifact_type}]--{prompt}")
                elif artifact_type == "synthesized_flow" or artifact_type == "flow":
                    # TODO. RGB Visualization of optical flow. (F,2,H,W)
                    artifact_value_RGB = flow_to_color(artifact_value) # BF,C,H,W (B=1)

                    frame_list = []
                    for frame in artifact_value_RGB:
                        frame = (frame.permute(1,2,0).float().detach().cpu().numpy()).astype(np.uint8).clip(0,255)
                        frame_list.append(Image.fromarray(frame))
                    
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(frame_list, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=f"[{artifact_type}]--{prompt}")

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log(
                        {
                            tracker_key: {f"images": image_artifacts, f"videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        del camera_flow_generator

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()


    # mangling
    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    # mangling
    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))


register("cogvideox-flovd-omsm", "lora", FloVDOMSMCogVideoXI2VLoraTrainer)


#--------------------------------------------------------------------------------------------------
# Extract function
def encode_text(prompt: str, components, device) -> torch.Tensor:
    prompt_token_ids = components.tokenizer(
        prompt,
        padding="max_length",
        max_length=components.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_token_ids = prompt_token_ids.input_ids
    prompt_embedding = components.text_encoder(prompt_token_ids.to(device))[0]
    return prompt_embedding

def encode_video(video: torch.Tensor, vae) -> torch.Tensor:
    # shape of input video: [B, C, F, H, W]
    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent

def decode_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    latents = 1 / vae.config.scaling_factor * latents

    frames = vae.decode(latents).sample
    return frames

def compute_optical_flow(raft, ctxt, trgt, raft_iter=20, chunk=2, only_forward=True):
    num_frames = ctxt.shape[0]
    chunk_size = (num_frames // chunk) + 1

    flow_f_list = []
    if not only_forward:
        flow_b_list = []
    for i in range(chunk):
        start = chunk_size * i
        end = chunk_size * (i+1)

        with torch.no_grad():
            flow_f = raft(ctxt[start:end], trgt[start:end], num_flow_updates=raft_iter)[-1]
            if not only_forward:
                flow_b = raft(trgt[start:end], ctxt[start:end], num_flow_updates=raft_iter)[-1]
        
        flow_f_list.append(flow_f)
        if not only_forward:
            flow_b_list.append(flow_b)
    
    flow_f = torch.cat(flow_f_list)
    if not only_forward:
        flow_b = torch.cat(flow_b_list)
    
    if not only_forward:
        return flow_f, flow_b
    else:
        return flow_f, None

def encode_flow(flow, vae, flow_scale_factor):
    # flow: BF,C,H,W
    # flow_scale_factor [sf_x, sf_y]
    assert flow.ndim == 4
    num_frames, _, height, width = flow.shape

    # Normalize optical flow
    # ndim: 4 -> 5
    flow = rearrange(flow, '(b f) c h w -> b f c h w', b=1)
    flow_norm = adaptive_normalize(flow, flow_scale_factor[0], flow_scale_factor[1])

    # ndim: 5 -> 4
    flow_norm = rearrange(flow_norm, 'b f c h w -> (b f) c h w', b=1)

    # Duplicate mean value for third channel
    num_frames, _, H, W = flow_norm.shape
    flow_norm_extended = torch.empty((num_frames, 3, height, width)).to(flow_norm)
    flow_norm_extended[:,:2] = flow_norm
    flow_norm_extended[:,-1:] = flow_norm.mean(dim=1, keepdim=True)
    flow_norm_extended = rearrange(flow_norm_extended, '(b f) c h w -> b c f h w', f=num_frames)

    return encode_video(flow_norm_extended, vae)

def decode_flow(flow_latent, vae, flow_scale_factor):
    flow_latent = flow_latent.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    flow_latent = 1 / vae.config.scaling_factor * flow_latent
    
    flow = vae.decode(flow_latent).sample # BCFHW

    # discard third channel (which is a mean value of f_x and f_y)
    flow = flow[:,:2].detach().clone()

    # Unnormalize optical flow
    flow = rearrange(flow, 'b c f h w -> b f c h w')
    flow = adaptive_unnormalize(flow, flow_scale_factor[0], flow_scale_factor[1])

    flow = rearrange(flow, 'b f c h w -> (b f) c h w')
    return flow # BF,C,H,W

def adaptive_normalize(flow, sf_x, sf_y):
    # x: BFCHW, optical flow
    assert flow.ndim == 5, 'Set the shape of the flow input as (B, F, C, H, W)'
    assert sf_x is not None and sf_y is not None
    b, f, c, h, w = flow.shape
    
    max_clip_x = math.sqrt(w/sf_x) * 1.0
    max_clip_y = math.sqrt(h/sf_y) * 1.0
    
    flow_norm = flow.detach().clone()
    flow_x = flow[:, :, 0].detach().clone()
    flow_y = flow[:, :, 1].detach().clone()
    
    flow_x_norm = torch.sign(flow_x) * torch.sqrt(torch.abs(flow_x)/sf_x + 1e-7)
    flow_y_norm = torch.sign(flow_y) * torch.sqrt(torch.abs(flow_y)/sf_y + 1e-7)

    flow_norm[:, :, 0] = torch.clamp(flow_x_norm, min=-max_clip_x, max=max_clip_x)
    flow_norm[:, :, 1] = torch.clamp(flow_y_norm, min=-max_clip_y, max=max_clip_y)

    return flow_norm


def adaptive_unnormalize(flow, sf_x, sf_y):
    # x: BFCHW, optical flow
    assert flow.ndim == 5, 'Set the shape of the flow input as (B, F, C, H, W)'
    assert sf_x is not None and sf_y is not None
    
    flow_orig = flow.detach().clone()
    flow_x = flow[:, :, 0].detach().clone()
    flow_y = flow[:, :, 1].detach().clone()
    
    flow_orig[:, :, 0] = torch.sign(flow_x) * sf_x * (flow_x**2 - 1e-7)
    flow_orig[:, :, 1] = torch.sign(flow_y) * sf_y * (flow_y**2 - 1e-7)
    
    return flow_orig

#--------------------------------------------------------------------------------------------------
