# Calibrion FT (Work In Progress)

🤖 **Agentic Fine-tuning Automation for Optimal Model Discovery**

Calibrion FT is an agent-driven fine-tuning library that semi-automatically orchestrates complex fine-tuning experiments to find the optimal generative model for your use case. The goal is to offer an agentic solution to bypass the traditional slow, manual hyperparameter searches and tedious experiment management.

## benefits

**Agentic**: Our AI agents:

- Were trained to produce optimal experiment configurations across multiple model variants
- Orchestrate fine-tuning jobs
- Help create evaluators together with the human expert
- Evaluate results and identify optimal hyperparameter combinations
- Handle failures gracefully and adapt experiment strategies in real-time

**Massive Time Savings**: What used to take weeks of manual work now happens autonomously

**Low Complexity**: Complex hyper-parameter configuration and searches that consumed a large part of a data scientist job can now be done by specialized agents, allowing the data scientist focus on high-level problem, solution success metrics.


## Manual Running of the pipeline

### Files

- `run_pipeline.py`: Orchestrates the full fine-tuning and evaluation pipeline.
- `step_1_run_ft_jobs.py`: Generates configurations and launches fine-tuning jobs.
- `step_2_update_experiments.py`: Updates experiment status and job completion.
- `step_3_eval_run_ft_models.py`: Runs fine-tuned models on the evaluation set.
- `step_4_run_evaluation.py`: Evaluates model outputs and logs results to Weights & Biases.
- `training_configs.py`: Stores lists of LLMs, batch sizes, and learning rate multipliers for experiments.

### Usage

To run the full pipeline, edit parameters as needed and execute:

```bash
python run_pipeline.py
```

You can control which steps to skip by passing the `skip_steps` argument to `run_pipeline()`.

### Arguments for `run_pipeline`

- `wandb_project` (str): Name of the Weights & Biases project for logging results.
- `dataset_version` (str): Dataset version to use for training and evaluation.
- `skip_steps` (list of int): List of step numbers to skip (e.g., `[1, 2]` skips steps 1 and 2).

### Outputs

- `_experiments.json`: Stores experiment configurations and job IDs.
- `_ft_models_eval_runs.json`: Contains evaluation results for fine-tuned models.

## Publishing to PyPI with uv

### 1. Init (first time only)

```bash
uv init --lib calibrion
```

As a result, the package files will be stored under `./calibrion`

### 2. Build

```bash
cd calibrion
uv build
```

### 3. Publish to PyPI

```bash
export UV_PUBLISH_TOKEN="pypi-AgEIcHlwaS5vcmcC..."
uv publish 
```

### 4. Bump version (before next release)

```bash
uv version patch   # 0.0.1 -> 0.0.2
uv version minor   # 0.1.0 -> 0.2.0
uv version major   # 1.0.0 -> 2.0.0
```
