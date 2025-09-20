import time
from pathlib import Path
import step_1_run_ft_jobs, step_2_update_experiments, step_3_eval_run_ft_models, step_4_run_evaluation
import training_configs
import json
import logging
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)

def run_pipeline(wandb_project: str = "sw-code-ai", dataset_version: str = "1.1.small", skip_steps: list[int] = None):
    """
    Run the complete fine-tuning and evaluation pipeline.
    
    Args:
        wandb_project: W&B project name
        dataset_version: Version of the dataset to use (must exist in versions.yaml)
        skip_steps: List of step numbers to skip (e.g. [1,2] skips steps 1 and 2)
    """
    skip_steps = skip_steps or []

    if 1 not in skip_steps:
        logger.info("Starting Step 1: Running fine-tuning jobs")
        try:
            experiments = step_1_run_ft_jobs.run_experiments(
                training_configurations=step_1_run_ft_jobs.generate_configurations(
                    dataset_version=dataset_version,
                    llms=training_configs.llms,
                    batch_sizes=training_configs.batch_sizes,
                    learning_rate_multipliers=training_configs.learning_rate_multipliers
                )
            )
            if experiments is None:
                logger.info("Pipeline aborted by user")
                return
        except Exception as e:
            logger.exception(f"Could not finish step 1: {str(e)}")
            raise

    if 2 not in skip_steps:
        logger.info("Starting Step 2: Waiting for fine-tuning jobs to complete")
        # Wait time before checking job completion in seconds
        waiting_time = 300
        while True:
            try:
                step_2_update_experiments.update_experiments()
                with open(Path(__file__).parent / "_experiments.json", 'r') as f:
                    experiments = json.load(f)
                if all('ft_model_id' in exp for exp in experiments.values()):
                    break
                minutes = waiting_time // 60
                seconds = waiting_time % 60
                logger.info(f"Not all jobs completed, waiting {minutes} minutes and {seconds} seconds...")
                time.sleep(waiting_time)
            except Exception as e:
                logger.exception(f"Error in Step 2: {str(e)}")
                raise
        logger.info("All fine-tuning jobs completed successfully")
    
    if 3 not in skip_steps:
        logger.info("Starting Step 3: Running fine-tuned models on evaluation set")
        try:
            step_3_eval_run_ft_models.eval_run_all_fted_models(dataset_version=dataset_version)
        except Exception as e:
            logger.exception(f"Could not finish step 3: {str(e)}")
            raise
    
    if 4 not in skip_steps:
        logger.info("Starting Step 4: Running evaluation and logging results to W&B")
        try:
            step_4_run_evaluation.evaluate_all_ft_models(wandb_project=wandb_project)
        except Exception as e:
            logger.exception(f"Could not finish step 4: {str(e)}")
            raise
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline(
        wandb_project="test-calibrion-ft",
        dataset_version="2.0.0",
        skip_steps=[1, 2, 3]
    )