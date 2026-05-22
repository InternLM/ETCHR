# Reinforcement Learning for ETCHR
This section of the codebase contains the implementation for Training Stage II: Reasoning Enhancement, which fine-tunes a question-conditioned image editor using Vision-Language Model (VLM) derived rewards.

## 🚀 Overview
The RL pipeline utilizes [Pref-GRPO](https://github.com/CodeGoat24/Pref-GRPO) to stabilize text-to-image reinforcement learning. It fine-tunes the Diffusion Transformer (DiT) of FLUX.2-klein-base-9B. 

To optimize the editor, we use a dual-reward mechanism driven by Qwen3-VL-8B-Instruct, which acts as both the downstream reasoning model ($\mathcal{M}$) and the Judge ($\mathcal{J}$).  

ETCHR calculates a convex sum of two complementary rewards: 
- **Editing Guidance Reward ($r_{guide}$)**: Measures downstream reasoning utility. It returns $1$ if the MLLM answers the question correctly given the edited image, and $0$ otherwise.  
- **Editing Correctness Reward ($r_{correct}$)**: A VLM-as-Judge assesses the edit in isolation. It returns $1$ if the edited image contains the valid visual information needed to answer the question, and $0$ otherwise.

The combined reward is computed as: $\mathcal{R}(i,i_{edit},q,a)=\alpha~r_{guide}+\beta~r_{correct}$. (By default, $\alpha=\beta=0.5$).

## 🛠️ Requirements & Environment
### Environment:
We use [Pref-GRPO](https://github.com/CodeGoat24/Pref-GRPO) as our codebase. Please follow the instructions from Pref-GRPO to prepare the environment.

1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/CodeGoat24/Pref-GRPO.git
cd Pref-GRPO
```

2. Install the training package:
```bash
conda create -n PrefGRPO python=3.12
conda activate PrefGRPO

bash env_setup.sh fastvideo

git clone https://github.com/mlfoundations/open_clip
cd open_clip
pip install -e .
cd ..

```

3. Install vLLM (for UnifiedReward-based rewards)
```bash
conda create -n vllm
conda activate vllm
pip install "vllm>=0.11.0"
pip install qwen-vl-utils==0.0.14
```

### Base Models:
- **Image Editor:** FLUX.2-klein-base-9B (Requires LoRA fine-tuned checkpoint from Stage I SFT).  
- **Reward Model/Judge:** Qwen3-VL-8B-Instruct.

## 🗄️ Data Preparation

The pipeline samples 10,000 instances across five task families. You can download the training parquet files from [https://huggingface.co/datasets/internlm/ETCHR-GRPO-10K](https://huggingface.co/datasets/internlm/ETCHR-GRPO-10K) and run ```prepare_data.py```, which organizes the training data from the parquet into the directory structure and format corresponding to ```RL/data/GRPO-10K.jsonl```.


## 💻 Training

### Data Preprocess
Modify the file path in ```Pref-GRPO/fastvideo/data_preprocess/preprocess_flux2_klein_edit.sh``` and run
```bash
cd Pref-GRPO
bash fastvideo/data_preprocess/preprocess_flux2_klein_edit.sh
```
### Start Training
First start a VLLM server to provide guidance and correctness reward.
```bash
bash scripts/launch_vllm.sh
```
Then, modify your vllm server information in ``` scripts/lora/lora_flux2_klein_edit_guidance_correctness.sh``` and start GRPO training
```bash
bash scripts/lora/lora_flux2_klein_edit_guidance_correctness.sh
```

After training, you can merge the lora weight by running ```merge_lora.py```