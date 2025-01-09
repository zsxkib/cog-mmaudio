# Training

## Overview

We have put a large emphasis on making training as fast as possible.
Consequently, some pre-processing steps are required.

Namely, before starting any training, we

1. Encode training audios into spectrograms and then with VAE into mean/std
2. Extract CLIP and synchronization features from videos
3. Extract CLIP features from text (captions)
4. Encode all extracted features into [MemoryMappedTensors](https://pytorch.org/tensordict/main/reference/generated/tensordict.MemoryMappedTensor.html) with [TensorDict](https://pytorch.org/tensordict/main/reference/tensordict.html)

**NOTE:** for maximum training speed (e.g., when training the base model with 2*H100s), you would need around 3~5 GB/s of random read speed. Spinning disks would not be able to catch up and most consumer-grade SSDs would struggle. In my experience, the best bet is to have a large enough system memory such that the OS can cache the data. This way, the data is read from RAM instead of disk.

The current training script does not support `_v2` training.

## Prerequisites

Install [av-benchmark](https://github.com/hkchengrex/av-benchmark). We use this library to automatically evaluate on the validation set during training, and on test set after training.
You will also need ffmpeg for video frames extraction. Note that `torchaudio` imposes a maximum version limit (`ffmpeg<7`). You can install it as follows:

```bash
conda install -c conda-forge 'ffmpeg<7'
```

Download the corresponding VAE (`v1-16.pth` for 16kHz training, and `v1-44.pth` for 44.1kHz training), vocoder models (`best_netG.pt` for 16kHz training; the vocoder for 44.1kHz training will be downloaded automatically), the [empty string encoding](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth), and Synchformer weights from [MODELS.md](https://github.com/hkchengrex/MMAudio/blob/main/docs/MODELS.md) place them in `ext_weights/`.

## Preparing Audio-Video-Text Features

We have prepared some example data in `training/example_videos`. 
Running the `training/extract_video_training_latents.py` script will extract the audio, video, and text features and save them as a `TensorDict` with a `.tsv` file containing metadata on disk.

To run this script, use the `torchrun` utility:

```bash
torchrun --standalone training/extract_video_training_latents.py
```

You can run this with multiple GPUs (with `--nproc_per_node=<n>`) to speed up extraction.
Check the top of the script to switch between 16kHz/44.1kHz extraction and data path definitions.

Arguments:

- `latent_dir` -- where intermediate latent outputs are saved. It is safe to delete this directory afterwards.
- `output_dir` -- where TensorDict and the metadata file are saved.

## Preparing Audio-Text Features

We have prepared some example data in `training/example_audios`. 
We first need to run `training/partition_clips` to partition each audio file into clips. 
Then, we run the `training/extract_audio_training_latents.py` script, which will extract the audio and text features and save them as a `TensorDict` with a `.tsv` file containing metadata on the disk.

To run this script:

```bash
python training/partition_clips.py
```

Arguments:

- `data_path` -- path to the audio files (`.flac` or `.wav`)
- `output_dir` -- path to the output `.csv` file
- `start` -- optional; useful when you need to run multiple processes to speed up processing -- this defines the beginning of the chunk to be processed
- `end` -- optional; useful when you need to run multiple processes to speed up processing -- this defines the end of the chunk to be processed

Then, run the `extract_audio_training_latents.py` with `torchrun`:

```bash
torchrun --standalone training/extract_audio_training_latents.py
```

You can run this with multiple GPUs (with `--nproc_per_node=<n>`) to speed up extraction.
Check the top of the script to switch between 16kHz/44.1kHz extraction.

Arguments:

- `data_dir` -- path to the audio files (`.flac` or `.wav`), same as the previous step
- `captions_tsv` -- path to the captions file, a csv file at least with columns `id` and `caption`
- `clips_tsv` -- path to the clips file, generated in the last step
- `latent_dir` -- where intermediate latent outputs are saved. It is safe to delete this directory afterwards.
- `output_dir` -- where TensorDict and the metadata file are saved.

The reference tsv files (with overlaps removed as mentioned in the paper) can be found [here](https://github.com/hkchengrex/MMAudio/releases/tag/v0.1). Note that `audioset_sl.tsv`, `bbcsound.tsv`, and `freesound.tsv` are subsets that are parts of WavCaps. These subsets might be smaller than the original datasets.

## Training

We use Distributed Data Parallel (DDP) for training.
First, specify the data path in `config/data/base.yaml`. If you used the default parameters in the scripts above to extract features for the example data, the `Example_video` and `Example_audio` items should already be correct.

To run training on the example data, use the following command:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 train.py exp_id=debug compile=False  debug=True example_train=True  batch_size=1
```

This will not train a useful model, but it will check if everything is set up correctly.

For full training on the base model with two GPUs, use the following command:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py exp_id=exp_1 model=small_16k
```

Any outputs from training will be stored in `output/<exp_id>`. 

More configuration options can be found in `config/base_config.yaml` and `config/train_config.yaml`.

## Checkpoints

Model checkpoints, including optimizer states and the latest EMA weights, are available here: https://huggingface.co/hkchengrex/MMAudio

---

Godspeed!
