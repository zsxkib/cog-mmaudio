# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import gc
from fractions import Fraction
from pathlib import Path
from datetime import datetime
import subprocess
import time
from typing import Optional

MODEL_CACHE = "cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/mmaudio"
MODEL_FILES = ["weights.tar", "ext_weights.tar", "cache.tar"]

import torch
import torchaudio
import numpy as np
from cog import BasePredictor, Input, Path as CogPath

# Enable TF32 for better performance on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import model utilities
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

from mmaudio.eval_utils import (
    ModelConfig,
    VideoInfo,
    all_model_cfg,
    generate,
    load_video,
    load_image,
    make_video,
    setup_eval_logging,
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)

    if ".tar" in dest:
        dest = os.path.dirname(dest)

    command = ["pget", "-vfx", url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        # Set up logging
        setup_eval_logging()
        
        # Add download logic at the start of setup
        for model_file in MODEL_FILES:
            url = WEIGHTS_BASE_URL + "/" + model_file
            dest_path = model_file

            dir_name = dest_path.replace(".tar", "")
            if os.path.exists(dir_name):
                print(f"[+] Directory {dir_name} already exists, skipping download")
                continue

            download_weights(url, dest_path)

        # Load the recommended large_44k_v2 model
        self.model_cfg: ModelConfig = all_model_cfg["large_44k_v2"]
        self.model_cfg.download_if_needed()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.seq_cfg = self.model_cfg.seq_cfg

        # Load network
        self.net: MMAudio = (
            get_my_mmaudio(self.model_cfg.model_name).to(self.device, self.dtype).eval()
        )
        self.net.load_weights(
            torch.load(
                self.model_cfg.model_path, map_location=self.device, weights_only=True
            )
        )
        print(f'Loaded weights from {self.model_cfg.model_path}')

        # Load feature utilities with need_vae_encoder=False for optimization
        self.feature_utils = (
            FeaturesUtils(
                tod_vae_ckpt=self.model_cfg.vae_path,
                synchformer_ckpt=self.model_cfg.synchformer_ckpt,
                enable_conditions=True,
                mode=self.model_cfg.mode,
                bigvgan_vocoder_ckpt=self.model_cfg.bigvgan_16k_path,
                need_vae_encoder=False,
            )
            .to(self.device, self.dtype)
            .eval()
        )

        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def predict(
        self,
        prompt: str = Input(description="Text prompt for generated audio", default=""),
        negative_prompt: str = Input(
            description="Negative prompt to avoid certain sounds", default="music"
        ),
        video: Optional[CogPath] = Input(
            description="Optional video file for video-to-audio generation",
            default=None,
        ),
        duration: float = Input(
            description="Duration of output in seconds", default=8.0, ge=1
        ),
        num_steps: int = Input(description="Number of inference steps", default=25),
        cfg_strength: float = Input(description="Guidance strength (CFG)", default=4.5, ge=1),
        seed: Optional[int] = Input(
            description="Random seed. Use -1 or leave blank to randomize the seed", default=None, ge=-1
        ),
        image: Optional[CogPath] = Input(
            description="Optional image file for image-to-audio generation (experimental)",
            default=None,
        ),
    ) -> CogPath:
        """
        If `video` is provided, generates audio that syncs with the given video and returns an MP4.
        If `image` is provided but no video, generates audio from the image and returns an MP4.
        If neither is provided, generates audio from text and returns a FLAC file.
        """

        # Cast paths to str if they're not None
        video_path = str(video) if video is not None else None
        image_path = str(image) if image is not None else None
        
        # Handle seed
        if seed is None or seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        rng = torch.Generator(device=self.device).manual_seed(seed)
        
        # Setup flow matching
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        # Update sequence configuration
        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(
            self.seq_cfg.latent_seq_len,
            self.seq_cfg.clip_seq_len,
            self.seq_cfg.sync_seq_len,
        )

        clip_frames = sync_frames = None
        video_info = None
        image_input = False
        
        # Process inputs - prioritize video over image
        if video_path:
            # Video-to-audio mode
            print(f"Processing video: {video_path}")
            video_info = load_video(video_path, duration)
            clip_frames = video_info.clip_frames.unsqueeze(0)
            sync_frames = video_info.sync_frames.unsqueeze(0)
            duration = video_info.duration_sec
            self.seq_cfg.duration = duration
            self.net.update_seq_lengths(
                self.seq_cfg.latent_seq_len,
                self.seq_cfg.clip_seq_len,
                self.seq_cfg.sync_seq_len,
            )
        elif image_path:
            # Image-to-audio mode
            print(f"Processing image: {image_path}")
            image_info = load_image(image_path)
            clip_frames = image_info.clip_frames.unsqueeze(0)
            sync_frames = image_info.sync_frames.unsqueeze(0)
            image_input = True
            video_info = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
        
        # Generate audio with no_grad
        with torch.inference_mode():
            audios = generate(
                clip_frames,
                sync_frames,
                [prompt],
                negative_text=[negative_prompt],
                feature_utils=self.feature_utils,
                net=self.net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength,
                image_input=image_input,
            )
        audio = audios.float().cpu()[0]

        current_time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if video_path or image_path:
            # Combine video/image and audio into an MP4
            video_save_path = self.output_dir / f"{current_time_string}.mp4"
            make_video(
                video_info,
                video_save_path,
                audio,
                sampling_rate=self.seq_cfg.sampling_rate,
            )
            gc.collect()
            return CogPath(video_save_path)
        else:
            # Just save audio as FLAC for text-to-audio
            audio_save_path = self.output_dir / f"{current_time_string}.flac"
            torchaudio.save(
                audio_save_path, audio.unsqueeze(0), self.seq_cfg.sampling_rate
            )
            gc.collect()
            return CogPath(audio_save_path)
