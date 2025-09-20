# experimentation

This folder contains scripts for running, tracking, and evaluating fine-tuning experiments.

## Files

- `run_pipeline.py`: Orchestrates the full fine-tuning and evaluation pipeline.
- `step_1_run_ft_jobs.py`: Generates configurations and launches fine-tuning jobs.
- `step_2_update_experiments.py`: Updates experiment status and job completion.
- `step_3_eval_run_ft_models.py`: Runs fine-tuned models on the evaluation set.
- `step_4_run_evaluation.py`: Evaluates model outputs and logs results to Weights & Biases.
- `training_configs.py`: Stores lists of LLMs, batch sizes, and learning rate multipliers for experiments.

## Usage

To run the full pipeline, edit parameters as needed and execute:

```bash
python run_pipeline.py
```

You can control which steps to skip by passing the `skip_steps` argument to `run_pipeline()`.

### Arguments for `run_pipeline`

- `wandb_project` (str): Name of the Weights & Biases project for logging results.
- `dataset_version` (str): Dataset version to use for training and evaluation.
- `skip_steps` (list of int): List of step numbers to skip (e.g., `[1, 2]` skips steps 1 and 2).

## Outputs

- `_experiments.json`: Stores experiment configurations and job IDs.
- `_ft_models_eval_runs.json`: Contains evaluation results for fine-tuned models.
