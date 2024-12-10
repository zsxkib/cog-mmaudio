import dataclasses
import logging
from pathlib import Path
from typing import Optional

import torch
from colorlog import ColoredFormatter
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder, StreamingMediaEncoder

from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio
from mmaudio.model.sequence_config import (CONFIG_16K, CONFIG_44K, SequenceConfig)
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.utils.download_utils import download_model_if_needed

log = logging.getLogger()


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str
    synchformer_ckpt: Path = Path('./ext_weights/synchformer_state_dict.pth')

    @property
    def seq_cfg(self) -> SequenceConfig:
        if self.mode == '16k':
            return CONFIG_16K
        elif self.mode == '44k':
            return CONFIG_44K

    def download_if_needed(self):
        download_model_if_needed(self.model_path)
        download_model_if_needed(self.vae_path)
        if self.bigvgan_16k_path is not None:
            download_model_if_needed(self.bigvgan_16k_path)
        download_model_if_needed(self.synchformer_ckpt)


small_16k = ModelConfig(model_name='small_16k',
                        model_path=Path('./weights/mmaudio_small_16k.pth'),
                        vae_path=Path('./ext_weights/v1-16.pth'),
                        bigvgan_16k_path=Path('./ext_weights/best_netG.pt'),
                        mode='16k')
small_44k = ModelConfig(model_name='small_44k',
                        model_path=Path('./weights/mmaudio_small_44k.pth'),
                        vae_path=Path('./ext_weights/v1-44.pth'),
                        bigvgan_16k_path=None,
                        mode='44k')
medium_44k = ModelConfig(model_name='medium_44k',
                         model_path=Path('./weights/mmaudio_medium_44k.pth'),
                         vae_path=Path('./ext_weights/v1-44.pth'),
                         bigvgan_16k_path=None,
                         mode='44k')
large_44k = ModelConfig(model_name='large_44k',
                        model_path=Path('./weights/mmaudio_large_44k.pth'),
                        vae_path=Path('./ext_weights/v1-44.pth'),
                        bigvgan_16k_path=None,
                        mode='44k')
large_44k_v2 = ModelConfig(model_name='large_44k_v2',
                           model_path=Path('./weights/mmaudio_large_44k_v2.pth'),
                           vae_path=Path('./ext_weights/v1-44.pth'),
                           bigvgan_16k_path=None,
                           mode='44k')
all_model_cfg: dict[str, ModelConfig] = {
    'small_16k': small_16k,
    'small_44k': small_44k,
    'medium_44k': medium_44k,
    'large_44k': large_44k,
    'large_44k_v2': large_44k_v2,
}


def generate(clip_video: Optional[torch.Tensor],
             sync_video: Optional[torch.Tensor],
             text: Optional[list[str]],
             *,
             negative_text: Optional[list[str]] = None,
             feature_utils: FeaturesUtils,
             net: MMAudio,
             fm: FlowMatching,
             rng: torch.Generator,
             cfg_strength: float):
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video, batch_size=bs)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video, batch_size=bs)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


def load_video(video_path: Path, duration_sec: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    _CLIP_SIZE = 384
    _CLIP_FPS = 8.0

    _SYNC_SIZE = 224
    _SYNC_FPS = 25.0

    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(_CLIP_FPS * duration_sec),
        frame_rate=_CLIP_FPS,
        format='rgb24',
    )
    reader.add_basic_video_stream(
        frames_per_chunk=int(_SYNC_FPS * duration_sec),
        frame_rate=_SYNC_FPS,
        format='rgb24',
    )

    reader.fill_buffer()
    data_chunk = reader.pop_chunks()
    clip_chunk = data_chunk[0]
    sync_chunk = data_chunk[1]
    assert clip_chunk is not None
    assert sync_chunk is not None

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

    if clip_length_sec < duration_sec:
        log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {clip_length_sec:.2f} sec')
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {sync_length_sec:.2f} sec')
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

    return clip_frames, sync_frames, duration_sec


def make_video(video_path: Path, output_path: Path, audio: torch.Tensor, sampling_rate: int,
               duration_sec: float):

    approx_max_length = int(duration_sec * 60)
    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=approx_max_length,
        format='rgb24',
    )
    reader.fill_buffer()
    video_chunk = reader.pop_chunks()[0]
    assert video_chunk is not None

    fps = int(reader.get_out_stream_info(0).frame_rate)
    if fps > 60:
        log.warning(f'This code supports only up to 60 fps, but the video has {fps} fps')
        log.warning(f'Just change the *60 above me')

    h, w = video_chunk.shape[-2:]
    video_chunk = video_chunk[:int(fps * duration_sec)]

    writer = StreamingMediaEncoder(output_path)
    writer.add_audio_stream(
        sample_rate=sampling_rate,
        num_channels=audio.shape[0],
        encoder='aac',  # 'flac' does not work for some reason?
    )
    writer.add_video_stream(frame_rate=fps,
                            width=w,
                            height=h,
                            format='rgb24',
                            encoder='libx264',
                            encoder_format='yuv420p')
    with writer.open():
        writer.write_audio_chunk(0, audio.float().transpose(0, 1))
        writer.write_video_chunk(1, video_chunk)
