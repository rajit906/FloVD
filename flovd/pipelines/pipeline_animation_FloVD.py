# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect

import pandas as pd
import torch
import PIL.Image

from typing import Callable, List, Optional, Union, Dict
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    _append_dims,
    tensor2vid,
    StableVideoDiffusionPipelineOutput
)

from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.models.guidance_preprocessor import GuidancePreprocessor
from cameractrl.models.unet import UNetSpatioTemporalConditionModelPoseCond
from cameractrl.models.guidance_preprocessor import adaptive_normalize, adaptive_unnormalize, flow_vis
from cameractrl.modules.SceneFlow.scene_flow_depth import unnormalize_intrinsic
from einops import rearrange, repeat

import pdb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableVideoDiffusionPipelinePoseCondWarping(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModelPoseCond,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        pose_encoder: CameraPoseEncoder,
        guidance_preprocessor: GuidancePreprocessor
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            pose_encoder=pose_encoder,
            guidance_preprocessor=guidance_preprocessor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance, do_mixed_classifier_free_guidance, do_resize_normalize):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values
        elif do_resize_normalize:
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0
            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_mixed_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For mixed classifier free guidance, we need to do three forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings, image_embeddings])
            
        elif do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        do_mixed_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_mixed_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do three forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents, image_latents])
            
        elif do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        do_mixed_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_mixed_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids])
        elif do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    
    @property
    def guidance_scale_fix(self):
        return self._guidance_scale_fix
    
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        guidance_3D_input: Dict,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fix_guidance_scale: float = 1.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        do_resize_normalize: bool = True,
        do_image_process: bool = False,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        use_adapter_classifier_free_guidance: bool = False,
        use_resolution_weighting: bool = False,
        guidance_log_dict_flow: dict = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        # device = image.device
        device = guidance_3D_input['condition_image'].device
        
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0
        do_mixed_classifier_free_guidance = fix_guidance_scale > 1.0
        if do_mixed_classifier_free_guidance:
            assert do_classifier_free_guidance, "Use mixed CFG only when using the original CFG."
            assert use_adapter_classifier_free_guidance, "Set use_adapter_classifier_free_guidance as True if you want to use mixed CFG."

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance, do_mixed_classifier_free_guidance=do_mixed_classifier_free_guidance, do_resize_normalize=do_resize_normalize)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        if do_image_process:
            image = self.image_processor.preprocess(image, height=height, width=width).to(image_embeddings.device)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance, do_mixed_classifier_free_guidance=do_mixed_classifier_free_guidance)
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            do_mixed_classifier_free_guidance=do_mixed_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )           # [bs, frame, c, h, w]

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)     # [bs, frame, 1, 1, 1]

        self._guidance_scale = guidance_scale
        
        guidance_scale_fix = torch.tensor([fix_guidance_scale]*num_frames).unsqueeze(0)
        guidance_scale_fix = guidance_scale_fix.to(device, latents.dtype)
        guidance_scale_fix = guidance_scale_fix.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale_fix = _append_dims(guidance_scale_fix, latents.ndim)     # [bs, frame, 1, 1, 1]
        self._guidance_scale_fix = guidance_scale_fix
        
        
        # 8. TODO.Preprocess the 3D guidance (Filtering, Mask concatenation, etc.)
        # 아 애매하다..
        # inference time에는 여기서 jittering 적용하면 안될거같은데.
        # 뒤에 mask 사용하니까, 여기선 disabled.
        use_masked_guidance_tag = self.guidance_preprocessor.use_masked_guidance
        self.guidance_preprocessor.use_masked_guidance = False
        guidance_3D, guidance_log_dict = self.guidance_preprocessor(**guidance_3D_input)
        self.guidance_preprocessor.use_masked_guidance = use_masked_guidance_tag
        
        
        #-----------------------------------------------------------------------------------------------
        """
            Flow Integration
        """
        if guidance_log_dict_flow is not None:
            flow_motion = guidance_log_dict_flow['sample_orig']
            
            static_mask = guidance_log_dict_flow['static']
            object_mask = 1. - static_mask # guidance_preprocessor가 masked guidance를 사용하면, segmentation과 obseved mask가 겹쳐버릴 수 있음. 조심하기.
            
            # Integrating camera and object motion flows
            # 모든 forward flow를 target view로 camera warping하기
            # batch 구성 : [K_t, E_t, f_t, d_0]
            #------------------------------------------------------------------------------------------------------------------------
            batch_for_warping = guidance_3D_input['structure_estimator_input']
            b_, f_ = batch_for_warping["target"]["intrinsics"].shape[:2]
            h_, w_ = flow_motion.shape[-2:]
            
            flow_motion_for_warpig = rearrange(flow_motion, 'b f c h w -> (b f) c h w')
            
            c2w_ctxt = repeat(batch_for_warping["context"]["extrinsics"][:,:1], "b t h w -> (b v t) h w", v=f_) # No need to apply inverse as it is an eye matrix.
            c2w_trgt = rearrange(torch.inverse(batch_for_warping["target"]["extrinsics"]), "b t h w -> (b t) h w")
            intrinsics_ctxt = unnormalize_intrinsic(repeat(batch_for_warping["context"]["intrinsics"][:,:1], "b t h w -> (b v t) h w", v=f_), size=(h_,w_))
            intrinsics_trgt = unnormalize_intrinsic(rearrange(batch_for_warping["target"]["intrinsics"], "b t h w -> (b t) h w"), size=(h_,w_))
            depth_ctxt = guidance_log_dict['depth_ctxt'] # bfchw
            
            # TODO.알고리즘 다시 한 번 체크하기.
            warped_displacement=None
            with torch.cuda.amp.autocast(enabled=False):
                warped_displacement = self.guidance_preprocessor.structure_estimator_inference.structure_estimation_network.warper.forward_warp_displacement(
                    depth1=repeat(depth_ctxt.squeeze(0), "b c h w -> (b f) c h w", f=f_),
                    flow1=flow_motion_for_warpig, 
                    transformation1=c2w_ctxt, 
                    transformation2=c2w_trgt, 
                    intrinsic1=intrinsics_ctxt, 
                    intrinsic2=None,
                )
            # warped_flow_motion = flow_motion_for_warpig + warped_displacement
            warped_flow_motion = warped_displacement # Use warped displacement for flow magnitude.
            warped_flow_motion = rearrange(warped_flow_motion, '(b f) c h w -> b f c h w', f=f_)
            
            
            # warped_flow_motion, _, _, flow_motion_flow = self.guidance_preprocessor.structure_estimator_inference.structure_estimation_network.warper.forward_warp(
            #     frame1=flow_motion_for_warpig, 
            #     mask1=None, 
            #     depth1=repeat(depth_ctxt.squeeze(0), "b c h w -> (b f) c h w", f=f_), 
            #     transformation1=c2w_ctxt, 
            #     transformation2=c2w_trgt, 
            #     intrinsic1=intrinsics_ctxt, 
            #     intrinsic2=None,
            #     is_image=False)
            
            # warped_flow_motion = rearrange(warped_flow_motion, '(b f) c h w -> b f c h w', f=f_)
            
            #------------------------------------------------------------------------------------------------------------------------
            flow_motion_orig = flow_motion.detach().clone()
            flow_motion = adaptive_normalize(flow_motion, sf_x=self.guidance_preprocessor.scale_factor[0], sf_y=self.guidance_preprocessor.scale_factor[1]) # bfchw
            warped_flow_motion = adaptive_normalize(warped_flow_motion, sf_x=self.guidance_preprocessor.scale_factor[0], sf_y=self.guidance_preprocessor.scale_factor[1]) # bfchw
            
            # guidance_3D = static_mask * guidance_3D[:,:,:2] + object_mask * flow_motion
            # guidance_3D = guidance_3D[:,:,:2] + object_mask * flow_motion
            guidance_3D = guidance_3D[:,:,:2] + object_mask * warped_flow_motion
            
            guidance_log_dict['static'] = guidance_log_dict_flow['static']
            tmp = adaptive_unnormalize(guidance_3D.detach().clone(), sf_x=self.guidance_preprocessor.scale_factor[0], sf_y=self.guidance_preprocessor.scale_factor[1])
            b_ = tmp.shape[0]
            tmp_vis = flow_vis(rearrange(tmp, "b f c h w -> (b f) c h w"), clip_flow=max(tmp.shape[-2:]))
            flow_motion_vis = flow_vis(rearrange(flow_motion_orig.cpu(), "b f c h w -> (b f) c h w"), clip_flow=max(flow_motion_orig.shape[-2:]))
            if warped_displacement is not None:
                warped_displacement_vis = flow_vis(warped_displacement.cpu(), clip_flow=max(warped_displacement.shape[-2:]))
            
            
            guidance_log_dict['guidance_3D_new_original'] = tmp
            guidance_log_dict['guidance_3D_new'] = rearrange(tmp_vis, "(b f) c h w -> b f c h w", b=b_)
            guidance_log_dict['motion_flow'] = rearrange(flow_motion_vis, "(b f) c h w -> b f c h w", b=b_)
            if warped_displacement is not None:
                guidance_log_dict['warped_displacement'] = rearrange(warped_displacement_vis, "(b f) c h w -> b f c h w", b=b_)
        
            if self.guidance_preprocessor.use_masked_guidance:
                assert self.guidance_preprocessor.structure_estimator_training.use_segmentation == True and self.guidance_preprocessor.structure_estimator_inference.use_segmentation == True, "Only support static mask (or object mask)"
                mask = guidance_log_dict['mask']
                
                guidance_3D_object_jittered = self.guidance_preprocessor.jitter_func(guidance_3D.detach().clone())
                guidance_3D = guidance_3D * mask + guidance_3D_object_jittered * (1 - mask)
                guidance_3D = torch.cat([guidance_3D, mask], dim=2) # [b, f, c, h, w]
                
                
                guidance_3D_vis = self.guidance_preprocessor.unnormalize_flow(guidance_3D[:,:,:2].detach().clone(), size=guidance_3D.shape[-2:])
                guidance_3D_vis = flow_vis(rearrange(guidance_3D_vis, "b f c h w -> (b f) c h w"), clip_flow=max(guidance_3D_vis.shape[-2:]))

                guidance_log_dict.update({
                    'guidance_3D_obj_jittered_vis': rearrange(guidance_3D_vis, '(b f) c h w -> b f c h w', b=b_),
                })

        # 9. Prepare pose features
        assert guidance_3D.ndim == 5                         # [b, f, c, h, w]
        pose_features = self.pose_encoder(guidance_3D)       # list of [b, c, f, h, w]
        # pdb.set_trace()
        # resolution_scale = [1.0]*8
        # if do_guidance_resolution_weighting:
        #     resolution_scale = [s * x.shape[-1] for s, x in zip(resolution_scale, pose_features)]
        #     resolution_scale = get_weighted_scale(resolution_scale, pose_features, orig_size=image_latents.shape[-2:])

        if use_adapter_classifier_free_guidance:
            pose_features_uncond = [torch.zeros_like(x) for x in pose_features] 
        else:
            pose_features_uncond = pose_features
        
        resolution_weight = [1.0] * len(pose_features)
        if use_resolution_weighting:
            resolution_weight = [latents.shape[-1] * w / x.shape[-1] for w, x in zip(resolution_weight, pose_features)]
        
        if do_mixed_classifier_free_guidance:
            # linear CFG for image condition and fixed CFG for flow condition
            pose_features = [torch.cat([x_uncond, x_uncond, x*w], dim=0) for x_uncond, x, w in zip(pose_features_uncond, pose_features, resolution_weight)]
        elif do_classifier_free_guidance:
            # linear CFG for image condition
            pose_features = [torch.cat([x_uncond, x*w], dim=0) for x_uncond, x, w in zip(pose_features_uncond, pose_features, resolution_weight)]
        else:
            # No CFG
            pose_features = pose_features
            
        # pose_features = [torch.cat([x_uncond, x*w], dim=0) for x_uncond, x, w in zip(pose_features_uncond, pose_features, resolution_weight)] if do_classifier_free_guidance else pose_features
        # pose_features = [torch.cat([x, x*s], dim=0) for x, s in zip(pose_features, resolution_scale)] if do_classifier_free_guidance else pose_features
        # pdb.set_trace()
        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                if do_mixed_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 3)
                elif do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    pose_features=pose_features,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_mixed_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_img_cond, noise_pred_full_cond = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_img_cond - noise_pred_uncond) + self.guidance_scale_fix * (noise_pred_full_cond - noise_pred_img_cond)
                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)        # [b, c, f, h, w]
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames, guidance_log_dict

        return StableVideoDiffusionPipelineOutput(frames=frames), guidance_log_dict
