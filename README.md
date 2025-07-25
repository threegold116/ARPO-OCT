<div align=center>
<img src="https://github.com/dongguanting/Tool-Star/blob/main/img/image.png" width="120px">
</div>



<h1 align="center"> ✨ Agentic Reinforced Policy Optimization</a></h1>


<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)]()
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)]()
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)]()
</div>

<!--
<p align="center">
🤗 <a href="https://huggingface.co/dongguanting/Qwen2.5-3B-ARPO" target="_blank">Qwen2.5-3B-ARPO</a> ｜
🤗 <a href="https://huggingface.co/dongguanting/Qwen2.5-7B-ARPO" target="_blank">Qwen2.5-7B-ARPO</a> ｜
🤗 <a href="https://huggingface.co/dongguanting/Llama3.1-8B-ARPO" target="_blank">Llama3.1-8B-ARPO</a> ｜
🤗 <a href="https://huggingface.co/dongguanting/Qwen3-8B-ARPO-DeepSearch" target="_blank">Qwen3-8B-ARPO-DeepSearch</a> ｜ 
🤗 <a href="https://huggingface.co/dongguanting/Qwen3-14B-ARPO-DeepSearch" target="_blank">Qwen3-14B-ARPO-DeepSearch</a> ｜
</p>
<p align="center">
🤗 <a href="https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K" target="_blank">ARPO-SFT-54K</a> ｜
🤗 <a href="https://huggingface.co/datasets/dongguanting/ARPO-RL-Reasoning-10K" target="_blank">ARPO-RL-Reasoning-10K</a>
🤗 <a href="https://huggingface.co/datasets/dongguanting/ARPO-RL-DeepSearch-1K" target="_blank">ARPO-RL-DeepSearch-1K</a>
</p>
-->



<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News

- **[July 25, 2025]**: The brief introduction of ARPO can be found on platforms like **[X](), [Zhihu]() and [Wechat]()**.
- **[July 25, 2025]**: 🔥 We released all our ARPO checkpoint（3B~14B） and datasets (SFT, RL and Inference). Checkout **[🤗ARPO Collection](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)** here. We will keep update it!
- **[July 25, 2025]**: 📄 Our paper is now available on **[arXiv]()** and **[Hugging Face]()** daily paper.
- **[July 25, 2025]**: 🚀 Full codebase released. ARPO supports multi-tool agentic RL for the Qwen2.5, Qwen3, and Llama3 series models. Our team has implemented extensive parallelization and memory optimization during tool calling. We welcome you to give it a try.


## :mag_right: Roadmap

Tool-star is still under development and there are many issues and room for improvement. We will continue to update. And we also sincerely welcome contributions on this open-source toolkit.
- [x] Release tiny LLM version (e.g. 0.5B, 1.5B)
- [x] Support larger parameter size LLM (e.g. 7B)
- [ ] Support More Reasoning Tools ---> will released in our future work.


## Table of Contents

- [Tool-Star](#🔧✨tool-star-empowering-llm-brained-multi-tool-reasoner-via-reinforcement-learning)
  - [Overall Performance](#-overall-performance)
- [Quick Start](#-quick-start-for-training)
  - [Cold-Start SFT Stage](#-cold-start-sft-stage)
    - [Environment Setup](#1-environment-setup)
    - [Fine-Tuning Model](#2-fine-tuning-model)
  - [Self-Critic RL Stage](#-self-critic-rl-stage)
    - [Environment Setup](#1-environment-setup-1)
    - [Vanilla RL Training](#2-vanilla-rl-training)
    - [Optional: Self-Critic DPO Training](#3-self-critic-dpo-training-optional)
  - [TIR Evaluation](#-tir-evaluation)
    - [Environment Setup](#1-environment-setup-2)
    - [LLM Service Deployment](#2-llm-service-deployment)
    - [Retriever Serving Deployment](#3-retriever-serving-deployment)
    - [Inference Your Model](#4-inference-your-model)
    - [Calculate Metrics](#5-calculate-metrics)
  - [Performance of Tool-Star Models](#-performance-of-tool-star-models)
- [Citation](#-citation)




## 💡 Overview


---

### 📊 Overall Performance




# 🏃 Quick Start for Training

## ❄️ Cold-Start SFT Stage

### 1. Environment Setup

In this step, we will describe how to perform a cold start for the SFT stage using the LLaMA Factory repository. First, set up the environment as follows:

```bash
# Clone the ARPO repository (which includes LLaMA-Factory)
git clone https://github.com/dongguanting/ARPO
cd ARPO/LLaMA-Factory

# Create a new conda environment
conda create -n sft python=3.10
conda activate sft

# Install dependencies
pip install -r requirements.txt
```

### 2. Fine-Tuning Model


1. Download your SFT dataset from [🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K) and place it in `LLaMA-Factory-main/data/final_sft_edition9.json`. Define the dataset in `dataset_info.json`.

2. Complete the path information in `LLaMA-Factory/arpo_train_sft/yaml`. The file content should be as follows:

```yaml
### model
model_name_or_path: your_model_path/Qwen3-14B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: ../examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset_dir: dataset_info
dataset: your_dataset
template: qwen
cutoff_len: 15000
max_samples: 1000000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: checkpoints/qwen
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 7.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

```

After completing the information, you can fine-tune the model using the following command:

```python
bash arpo_train_sft/sft_train.sh
```

---

## 🔥 Self-Critic RL Stage

In this step, we will load the cold-start data for GRPO training. We reference the [ReCall](https://github.com/Agent-RL/ReCall) and [VERL](https://github.com/volcengine/verl) frameworks for RL training.


### 1. Environment Setup

 you can install our additional environment as follow: 

```bash
#create env
conda create -n toolstar python==3.10
conda activate toolstar

# install torch & flash-atten
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# install RL basic env
cd Tool_Star_RL
pip3 install -e .

# This is our RL env freeze file. You can install it as a supplement or use it for checking.
pip install -r ./Tool-Star-main/requirements.txt

```
Please refer to [requirements.txt](https://github.com/dongguanting/Tool-Star/blob/main/requirements.txt) carefully. It is important to note that **vLLM<= 0.6.3 and torch==2.4.0 (seem versions will not work.)**. You can also install a compatible flash_attention package from [here](https://github.com/Dao-AILab/flash-attention/releases).

If you encounter ray or other RL environment issues, we **highly recommend that you first try to run the RL training code for [ReCall](https://github.com/Agent-RL/ReCall/tree/re-search) or [Verl](https://github.com/volcengine/verl) successfully**, then further aligning with our [requirements.txt](https://github.com/dongguanting/Tool-Star/blob/main/requirements.txt).



### 2. Vanilla RL Training

Our training framework is based on [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). The training scripts can be found under `scripts/train`. First, you need to complete the information in `scripts/train/run_tool_star.sh`, 
we have provided both [train parquet](https://huggingface.co/datasets/dongguanting/Multi-Tool-RL-10K) and [test parquet](https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet) for RL:

```bash
export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path {your_actor_model_path} \
    --project_name {your_project_name} \
    --experiment_name {your_experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --wandb_api_key {your_wandb_api_key} \
    --save_path {your_save_path} \
    --train_files {path_to_train_file}/grpo_mix_train_shuffle.parquet \
    --test_files {path_to_test_file}/grpo_mix_test.parquet
```

Since the rollout process involves Bing web search calls, please configure the `deep_search_snippet()` function in `/src/verl/verl/workers/rollout/vllm_rollout/web_search/web_search_main.py` with your search API:

```python
def deep_search_snippet(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key, 
        bing_endpoint=bing_endpoint, 
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
```

Replace `bing_subscription_key`, `bing_endpoint`, and `api_base_url` with your own values. Various web search modes are provided in this file for you to choose from.

You can then run the following script to start training:

```bash
cd ./Tool_Star_RL/scripts/train/
bash run_tool_star.sh
```

For the core code of the rollout process, please refer to `/src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`, and for the reward calculation part, refer to `/Tool_Star_RL/src/verl/verl/utils/reward_score`. You can modify them according to your needs.

For the trained RL checkpoint, you can follow the code below to convert the weights to Hugging Face format：
```bash
# Merge RL weights and save in the same path.
python /Tool_Star_RL/model_merger.py \
    --local_dir /{your_checkpoint_path}/global_step_{your_RL_step}/actor/ \
```


---

## ✅ ARPO Evaluation

If you have already trained a model, you can refer to the following process for TIR capability evaluation. Of course, you can also download our checkpoint **[🤗Tool-Star-Qwen-3B](https://huggingface.co/dongguanting/Tool-Star-Qwen-3B)** for directly testing.
This guide walks you through setting up two separate environments:
- One for **vLLM inference service** (`vllm_env`)
- One for **evaluation pipeline** (`evaluation`)

### 1. Setup vLLM Inference Environment

```bash
# Step into the vllm_scripts directory
cd evaluation/vllm_scripts

# Create a dedicated conda environment for vLLM
conda create -n vllm_env python=3.10
conda activate vllm_env

# Install dependencies (edit as needed)
pip install -r requirements.txt
```

Edit the following launch scripts with your own model paths and names:

In `vllm_launch_reasoning_model_cuda4-7.sh`:
```bash
MODEL_PATH="<path/to/your/reasoning_model_checkpoint>"
MODEL_NAME="your_model_name"
```

For summarization models (choose one):
```bash
MODEL_PATH="<path/to/your/summarization_model_checkpoint>"
MODEL_NAME="your_summarization_model_name"
```

Launch the vLLM services:
```bash
# Start the reasoning model
bash vllm_launch_reasoning_model_cuda4-7.sh

# Start the summarization model (choose one)
bash vllm_launch_summarize_model_cuda0-3_<your_model>.sh
```

---

### 2. Setup Evaluation Environment

```bash
# Create a separate environment for evaluation
conda create -n evaluation python=3.10
conda activate evaluation

# Install required packages
cd evaluation
pip install -r requirements.txt
```

---

### 3. Configure and Run Evaluation

Edit the `infer_local_sds.sh` script with the following values:

```bash
# Activate your Conda environment manually if 'conda' is not available in shell
source < /path/to/your/conda >/bin/activate
conda activate < your env name >

# Datasets to evaluate — uncomment the ones you want to include:
# Options: aime24, aime25, math500, gsm8k, math, webwalker, hotpotqa, 2wiki, bamboogle, musique, hle, gaia, SimpleQA, xbench
data_names=(
    "hle"
    "gaia"
)

# Required parameters to update:
EXP_NAME="<your_exp_name>"                   # Name of this experiment run
MODEL_PATH="<your_model_path>"               # Path to the reasoning model
OUTPUT_PATH="<your_output_path>"             # Directory to save outputs
CONDA_PATH="<your_conda_path>"               # Path to your Conda installation
CONDA_ENV="<your_env_name>"                  # Name of your Conda environment
BING_API_KEY="<your_bing_search_api_key>"    # Bing Search API key
BING_ZONE="<your_bing_zone>"                 # Bing API zone
SUMM_MODEL_PATH="<your_summarization_model_path>"  # Path to summarization model checkpoints
```

Run the evaluation:
```bash
bash evaluation/infer_local_sds.sh
```

> 🔸 For Chinese datasets like `xbench`, use `infer_local_sds_cn.sh` instead.


### 4. Calculate Metrics

After generating inference results, you can use a large model like **Qwen2.5-72B-Instruct** to evaluate them with more powerful understanding capabilities.

First, use the vLLM environment to start the evaluation model:

```bash
bash evaluation/deploy_qwen2.5_72B_instruct.sh
```

In that script, make sure to update the `vllm serve` command with your own model path:

```bash
# Activate your Conda environment manually if 'conda' is not available in shell
source < /path/to/your/conda >/bin/activate
conda activate < your env name >

vllm serve <your_model_path> \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 4 \
  --gpu-memory-utilization 0.75 \
  --quantization gptq \
  --port 8001
```

Before running the evaluation script, update the following line in `evaluate_passk.sh` to specify the output directory:

```bash
OUTPUT_DIR="<your_result_directory>"
```

Then, run the evaluation script to calculate metrics:

```bash
bash evaluation/evaluate_passk.sh
```
---

## 📄 Performance of ARPO Models

We present the results of our ARPO model checkpoints with sizes 8B and 14B, all based on the Qwen2.5-Instruct series. The results of **“Self-Critic-RL”** setting correspond to our series of 🤗 open-source huggingface model checkpoints.

## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
@article{dong2025toolstar,
  author       = {Guanting Dong and
                  Yifei Chen and
                  Xiaoxi Li and
                  Jiajie Jin and
                  Hongjin Qian and
                  Yutao Zhu and
                  Hangyu Mao and
                  Guorui Zhou and
                  Zhicheng Dou and
                  Ji{-}Rong Wen},
  title        = {Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement
                  Learning},
  journal      = {CoRR},
  volume       = {abs/2505.16410},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.16410},
  doi          = {10.48550/ARXIV.2505.16410},
  eprinttype    = {arXiv},
  eprint       = {2505.16410},
  timestamp    = {Thu, 26 Jun 2025 07:49:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-16410.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## 🤝 Acknowledge

This training implementation builds upon [Llama Factory](https://github.com/hiyouga/LLaMA-Factory), [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). For evaluation, we rely on [WebThinker](https://github.com/RUC-NLPIR/WebThinker), [Search-o1](https://github.com/sunnynexus/Search-o1), and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). The Python interpreter design references [ToRA](https://github.com/microsoft/ToRA) and [ToRL](https://github.com/GAIR-NLP/ToRL), while our models are trained using [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/). We express our sincere gratitude to these projects for their invaluable contributions to the open-source community. 


## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [dongguanting@ruc.edu.cn](dongguanting@ruc.edu.cn).


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dongguanting/Tool-Star&type=Date)](https://www.star-history.com/#dongguanting/Tool-Star&Date)
