# [Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis](https://hkchengrex.github.io/MMAudio)

[Ho Kei Cheng](https://hkchengrex.github.io/), [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Alexander Schwing](https://www.alexander-schwing.de/), [Yuki Mitsufuji](https://www.yukimitsufuji.com/)

University of Illinois Urbana-Champaign, Sony AI, and Sony Group Corporation


[[Paper (being prepared)]](https://hkchengrex.github.io/MMAudio) [[Project Page]](https://hkchengrex.github.io/MMAudio)


**Note: This repository is still under construction. Single example inference should work as expected. Training code will be added. Code is subject to non-backward-compatible changes.**

## Highlight

MMAudio generates synchronized audio given video and/or text inputs.
Our key innovation is multimodal joint training which allows training on a wide range of audio-visual and audio-text datasets.
Moreover, a synchronization module is used to align the generated audio with the video frames.


## Results


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

**Download the pretrained models:**

Expected directory structure (full):
```
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

Expected directory structure (minimal, for the recommended model only):
```
MMAudio
├── ext_weights
│   ├── synchformer_state_dict.pth
│   └── v1-44.pth
├── weights
│   └── mmaudio_large_44k.pth
└── ...
```

## Demo

By default, these scripts uses the `large_44k` model. 
In our experiments, it only uses around 6GB of GPU memory (in 16-bit mode) which should fit in most modern GPUs.

### Command-line interface

With `demo.py`
```bash
python demo.py --duration=8 --video=<path to video> --prompt "your prompt" 
```
See the file for more options.
Simply omit the `--video` option for text-to-audio synthesis.


### Gradio interface

```
python gradio_demo.py
```

## Training
Work in progress.