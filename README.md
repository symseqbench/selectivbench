# SelectivBench: Evaluating Selectivity in Linear Recurrent Models

This repository contains source code and simulation scripts to perform the experiments presented in the paper titled "Dissecting Linear Recurrent Models: How Different Gating Strategies Drive Selectivity and Generalization". 2026, arXiv preprint.

This work introduces SelectivBench, a benchmark suite that extends [SymSeqBench](https://arxiv.org/abs/2512.24977) to evaluate small-scale language models on tasks of increasing complexity, targeting NLP-relevant capabilities such as selective information routing and generalization.

**Note:** The code in symseqbench is based on older versions of [SeqBench](https://github.com/symseqbench/SeqBench) and [SymSeq](https://github.com/symseqbench/SymSeq) and will be updated to the latest versions soon.

## Installation

1. Install uv:

```bash
pip install uv

# or

pipx install uv
```

2. Clone the repository:

```bash
git clone https://github.com/symseqbench/selectivbench.git
cd selectivbench
```

3. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
uv pip install -e ".[dev]"
```

## Running Experiments

### Exemplary training command

```bash
python train.py \
  --batch_size 64 \
  --warmup_steps 1000 \
  --lr 0.001 \
  --train_gap_min 0.1 \
  --classify \
  --base_set one_hot \
  --save_model True \
  --run_main \
  --hash_save \
  --test_seqlen_max 200 \
  --grad_acc -1 \
  --grammar_scale True \
  --amb 24 \
  --d_state 128 \
  --feat_size 1024 \
  --layers 8 \
  --model transformer \
  --seed 3 \
  --train_gap_max 0.1 \
  --train_noise 0 \
  --train_seqlen_max 200
```

**Note:** the first run of the above command generate the sequence data in the folder `Data/SeqBench`.
It takes about half an hour to generate the data. For debugging purposes, you could consider changing the number
of generated sequences using the `--dataset_size` argument.

### Retrieve and Use W&B Sweeps

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and hyperparameter sweeps.

Hyperparameter sweep configuration files for the four benchmark tasks are located in the `wb_configs/` directory:

```
wb_configs/
├── task1_complexity.yaml
├── task1_model_size.yaml
├── task2.yaml
├── task3.yaml
└── task4.yaml
```

Each YAML file defines a W&B sweep for a specific task.

### Launch a Sweep

First, log in to W&B and execute:

```bash
python generate_sweep_id.py wb_configs/taskx.yaml
```

The output will include a **Sweep ID** (e.g., `username/project/sweep_id`). Use this ID in the next step.

### Run the Agent

Now execute the agent to run the experiments:

```bash
wandb agent username/project/sweep_id
```

You can run multiple agents in parallel across machines to accelerate experimentation.

## Citation

If you use SelectivBench code base or its derived tasks from SymSeqBench, please cite the following paper:

```bibtex
Bouhadjar, Y., Fabre, M., Schmidt, F & Neftci, E (2026). Dissecting Linear Recurrent Models: How Different Gating Strategies Drive Selectivity and Generalization. arXiv preprint.
```

## Project Structure

```
SelectivBench/
├── symseqbench/      # Core library for the SymSeqBench library
├── wb_configs/       # W&B sweep config files for each benchmark task
├── models/           # Models implementation
├── train.py          # Training and evaluation script
└── README.md
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### symseqbench Directory

The `symseqbench/` directory contains code derived from both [SeqBench](https://github.com/symseqbench/SeqBench) and [SymSeq](https://github.com/symseqbench/SymSeq) projects, both of which are licensed under the MIT License. The code in this directory is therefore also licensed under the MIT License. See the [symseqbench/LICENSE](symseqbench/LICENSE) file for details.

**Additional Note:** The `dataset/shd_ssc.py` file contains code from the Idiap Research Institute and is licensed under BSD-3-Clause. See the file header for details.
