# Supervised Fine-tuning for ETCHR
This section of the codebase contains the implementation for Training Stage I: Reasoning Imitation, which fine-tunes a passive instruction-following image editor (built on [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)) into an autonomous, question-conditioned visual reasoning assistant.

## 🚀 Overview
The SFT pipeline is built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/). It fine-tunes the Diffusion Transformer (DiT) of FLUX.2-klein-base-9B. 


## 🛠️ Requirements & Environment
### Environment:
We use [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/) as our codebase. Please follow the instructions from DiffSynth-Studio to prepare the environment.


Install from source (recommended):

```
cd DiffSynth-Studio
pip install -e .
```


### Base Models:
- **Image Editor:** [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)

## 🗄️ Data Preparation
The SFT corpus consists of curated question-edit trajectories, including five categories:

- **Fine-grained Perception:** $V^*$-GQA & $V^*$-COCO with rendered bounding boxes.
- **Chart Understanding**: RefChartQA with bounding box overlays.
- **Logic Reasoning:** Synthetic maze topology paired with overlaid correct traversal paths.  
- **Jigsaw Reasoning:** Sourced from Spatial-SSRL, mapping scrambled inputs to fully restored image grids.  
- **3D Understanding:** Extracted from DL3DV-10K camera extrinsics to simulate realistic viewpoint transitions.

You can download the training parquet files from [https://huggingface.co/datasets/internlm/ETCHR-SFT-400K](https://huggingface.co/datasets/BeichenZhang/ETCHR-SFT-400K) and run ```SFT/prepare_data.py```, which organizes the training data from the parquet into the directory structure and format corresponding to ```SFT/data/sft_data.csv```.




## 💻 Training

You can start SFT training by running ```ft_klein.sh```

After training, you can merge the lora weight by running ```merge_lora.py```
