import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import json
import random

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
    load_binary_mask_compressed,
)

import pdb

if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class I2VFlowDataset(Dataset):
    """
    A dataset class for (image,flow)-to-video generation or image-to-flow_video that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(
        self, 
        max_num_frames: int, 
        height: int, 
        width: int, 
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        *args, 
        **kwargs
    ) -> None:
        data_root = Path(data_root)
        metadata_path = data_root / "metadata_revised.jsonl"
        assert metadata_path.is_file(), "For this dataset type, you need metadata.jsonl in the root path"
        
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

        self.prompts = [x["prompt"] for x in metadata]
        if 'curated' in str(data_root).lower():
            self.prompt_embeddings = [data_root / "prompt_embeddings" / (x["hash_code"] + '.safetensors') for x in metadata]
        else:
            self.prompt_embeddings = [data_root / "prompt_embeddings_revised" / (x["hash_code"] + '.safetensors') for x in metadata]
        self.videos = [data_root / "video_latent" / "x".join(str(x) for x in trainer.args.train_resolution) / (x["hash_code"] + '.safetensors') for x in metadata]
        self.images = [data_root / "first_frames" / (x["hash_code"] + '.png') for x in metadata]
        self.flows = [data_root / "flow_direct_f_latent" / (x["hash_code"] + '.safetensors') for x in metadata]


        # data_root = Path(data_root)
        # self.prompts = load_prompts(data_root / caption_column)
        # self.videos = load_videos(data_root / video_column)

        self.trainer = trainer

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        
        # Check if number of prompts matches number of videos and images
        if not (len(self.videos) == len(self.prompts) == len(self.images) == len(self.flows)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=}, {len(self.images)=} and {len(self.flows)=}. Please ensure that the number of caption prompts, videos and images match in your dataset."
            )

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms
        
        self.length = len(self.videos)

        print(f"Dataset size: {self.length}")

    def __len__(self) -> int:
        return self.length
    
    def load_data_pair(self, index):
        # prompt = self.prompts[index]
        prompt_embedding_path = self.prompt_embeddings[index]
        encoded_video_path = self.videos[index]
        encoded_flow_path = self.flows[index]
        # mask_path = self.masks[index]
        # image_path = self.images[index]
        # train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        
        prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"] 
        encoded_video = load_file(encoded_video_path)["encoded_video"] # CFHW
        encoded_flow = load_file(encoded_flow_path)["encoded_flow_f"] # CFHW
        
        return prompt_embedding, encoded_video, encoded_flow
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                prompt_embedding, encoded_video, encoded_flow = self.load_data_pair(index)
                break
            except Exception as e:
                print(f"Error loading {self.prompt_embeddings[index]}: {str(e)}")
                index = random.randint(0, self.length - 1)
            
        image_path = self.images[index]
        prompt = self.prompts[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        
        _, image = self.preprocess(None, image_path)
        image = self.image_transform(image)
        

        # shape of encoded_video: [C, F, H, W]
        # shape and scale of image: [C, H, W], [-1,1]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_flow": encoded_flow,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)