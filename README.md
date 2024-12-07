# [Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis](https://hkchengrex.github.io/MMAudio)

[Ho Kei Cheng](https://hkchengrex.github.io/), [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Alexander Schwing](https://www.alexander-schwing.de/), [Yuki Mitsufuji](https://www.yukimitsufuji.com/)

University of Illinois Urbana-Champaign, Sony AI, and Sony Group Corporation


[[Paper (being prepared)]](https://hkchengrex.github.io/MMAudio) [[Project Page]](https://hkchengrex.github.io/MMAudio)


**Note: This repository is still under construction. Single-example inference should work as expected. The training code will be added. Code is subject to non-backward-compatible changes.**

## Highlight

MMAudio generates synchronized audio given video and/or text inputs.
Our key innovation is multimodal joint training which allows training on a wide range of audio-visual and audio-text datasets.
Moreover, a synchronization module aligns the generated audio with the video frames.


## Results

(Videos from MovieGen/Hunyuan Video/VGGSound; audio from our algorithm)

https://github.com/user-attachments/assets/29230d4e-21c1-4cf8-a221-c28f2af6d0ca

For more results, go to https://hkchengrex.com/MMAudio/video_main.html.

## Installation

We have only tested this on Ubuntu.

### Prerequisites

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

- Python 3.8+
- PyTorch **2.5.1+** and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/)
- ffmpeg<7 ([this is required by torchaudio](https://pytorch.org/audio/master/installation.html#optional-dependencies), you can install it in a miniforge environment with `conda install -c conda-forge 'ffmpeg<7'`)

**Clone our repository:**

```bash
git clone https://github.com/hkchengrex/MMAudio.git
```

**Install with pip:**

```bash
cd MMAudio
pip install -e .
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)

**Pretrained models:**

The models will be downloaded automatically when you run the demo script. MD5 checksums are provided in `mmaudio/utils/download_utils.py`

| Model    | Download link | File size |
| -------- | ------- | ------- |
| Flow prediction network, small 16kHz | <a href="https://databank.illinois.edu/datafiles/k6jve/download" download="mmaudio_small_16k.pth">mmaudio_small_16k.pth</a> | 601M |
| Flow prediction network, small 44.1kHz | <a href="https://databank.illinois.edu/datafiles/864ya/download" download="mmaudio_small_44k.pth">mmaudio_small_44k.pth</a> | 601M |
| Flow prediction network, medium 44.1kHz | <a href="https://databank.illinois.edu/datafiles/pa94t/download" download="mmaudio_medium_44k.pth">mmaudio_medium_44k.pth</a> | 2.4G |
| Flow prediction network, large 44.1kHz **(recommended)** | <a href="https://databank.illinois.edu/datafiles/4jx76/download" download="mmaudio_large_44k.pth">mmaudio_large_44k.pth</a> | 3.9G |
| 16kHz VAE | <a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth">v1-16.pth</a> | 655M |
| 16kHz BigVGAN vocoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt">best_netG.pt</a> | 429M |
| 44.1kHz VAE |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth">v1-44.pth</a> | 1.2G | 
| Synchformer visual encoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth">synchformer_state_dict.pth</a> | 907M |

The 44.1kHz vocoder will be downloaded automatically.

The expected directory structure (full):

```bash
MMAudio
├── ext_weights
│   ├── best_netG.pt
│   ├── synchformer_state_dict.pth
│   ├── v1-16.pth
│   └── v1-44.pth
├── weights
│   ├── mmaudio_small_16k.pth
│   ├── mmaudio_small_44k.pth
│   ├── mmaudio_medium_44k.pth
│   └── mmaudio_large_44k.pth
└── ...
```

The expected directory structure (minimal, for the recommended model only):

```bash
MMAudio
├── ext_weights
│   ├── synchformer_state_dict.pth
│   └── v1-44.pth
├── weights
│   └── mmaudio_large_44k.pth
└── ...
```

## Demo

By default, these scripts use the `large_44k` model. 
In our experiments, inference only takes around 6GB of GPU memory (in 16-bit mode) which should fit in most modern GPUs.

### Command-line interface

With `demo.py`
```bash
python demo.py --duration=8 --video=<path to video> --prompt "your prompt" 
```
See the file for more options.
Simply omit the `--video` option for text-to-audio synthesis.
The default output (and training) duration is 8 seconds. Longer/shorter durations could also work, but a large deviation from the training duration may result in a lower quality.


### Gradio interface

Supports video-to-audio and text-to-audio synthesis.

```
python gradio_demo.py
```

### Known limitations

1. The model sometimes generates undesired unintelligible human speech-like sounds
2. The model sometimes generates undesired background music
3. The model struggles with unfamiliar concepts, e.g., it can generate "gunfires" but not "RPG firing".

We believe all of these three limitations can be addressed with more high-quality training data.

## Training
Work in progress.

## Evaluation
Work in progress.
