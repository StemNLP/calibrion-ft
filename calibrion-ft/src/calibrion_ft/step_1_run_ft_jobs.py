import json
import shortuuid
from pathlib import Path
from openai import OpenAI
from finetuning import run_finetuning
import logging
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)

with open('secrets/openai_api_key.json', 'r') as f:
    credentials = json.load(f)

client = OpenAI(
  api_key=credentials.get('openai_api_key'),
)

# TODO Test if all combinations are generated correctly
def generate_configurations(dataset_version: str,
                          llms,
                          batch_sizes,
                          learning_rate_multipliers):
    """
    Generate all hyperparameter configurations for fine-tuning experiments.

    Args:
        dataset_version (str): Version of the dataset to use
        llms (list): List of LLM model names to be used for fine-tuning.
        batch_sizes (list): List of batch sizes to be used in the experiments.
        learning_rate_multipliers (list): List of learning rate multipliers to be used in the experiments.

    Returns:
        list: A list of dictionaries, each containing a configuration for a fine-tuning experiment.
    """
    from .dataset_config import get_dataset_files
    
    logger.info("Generating configurations for fine-tuning experiments...")
    configs = []
    
    train_file, test_file, train_file_id, test_file_id = get_dataset_files(dataset_version)
    
    # Loop to add the default config (None hyperparams) for each model
    for llm in llms:
        configs.append({
            "model": llm,
            "training_file": train_file,
            "training_file_oai_id": train_file_id,
            "test_file": test_file,
            "test_file_oai_id": test_file_id,
            "hyperparameters": None
        })
    
    # Loop to generate all hyperparam combinations/configurations 
    for llm in llms:
        for batch_size in batch_sizes:
            for lr_mult in learning_rate_multipliers:
                config = {
                    "model": llm,
                    "training_file": train_file,
                    "training_file_oai_id": train_file_id,
                    "test_file": test_file,
                    "test_file_oai_id": test_file_id,
                    "hyperparameters": {
                        "batch_size": batch_size,
                        "learning_rate_multiplier": lr_mult,
                        "n_epochs": 4
                    }
                }
                configs.append(config)
    return configs

def run_experiments(training_configurations):
    """
    Run fine-tuning experiments based on the provided configurations.
    Args:
        training_configurations (list): List of dictionaries containing configurations for fine-tuning.

        Returns:
            dict: A dictionary where each key is a unique identifier for an experiment,
                  and each value is a dictionary containing the experiment details.
                    Example:
                    {
                        "VJGHLBfagGsopVwsAG48ND": {
                            "model": "gpt-4.1-mini-2025-04-14",
                            "training_file_oai_id": "file-G8tstQfCKpgCcE8mzzoxrC",
                            "training_file": "file-views90-train",
                            "training_file": "file-views90-val",
                            "hyperparameters": null,
                            "ft_job_id": "ftjob-42a982ad"
                        },
                        "gWKFh54X2xGShKpsh5G934": {
                            "model": "gpt-4.1-mini-2025-04-14",
                            "training_file_oai_id": "file-views18425-train",
                            "training_file": "file-views18425-train",
                            "training_file": "file-views18425-val",
                            "hyperparameters": null,
                            "ft_job_id": "ftjob-5a2ef1e8"
                        }
                    }
    """
    experiments = {}
    
    logger.warning(f"This will run \"{len(training_configurations)}\" experiments which can become costly.")
    user_input = input(f"Do you want to continue experiments? (y/N): ")
    if user_input.lower() != 'y':
        logger.info("Aborting experiment run.")
        return None
    logger.info("Proceeding with experiments...")
    
    for config in training_configurations:
        # Validate required fields
        if not all(key in config for key in ["model", "training_file", "training_file_oai_id", "hyperparameters"]):
            logger.error(f"Missing required fields in config: {config}. Skipping this configuration.")
            continue
            
        method_config = None if config["hyperparameters"] is None else {
            "type": "supervised",
            "supervised": {
                "hyperparameters": config["hyperparameters"]
            }
        }
        
        try:
            response = run_finetuning(
                model=config["model"],
                training_file=config["training_file_oai_id"],
                ft_method_config=method_config
            )
            
            # Generate unique experiment ID
            experiment_id = str(shortuuid.uuid())
            
            # Store experiment config with UUID as key
            experiments[experiment_id] = {
                "model": config["model"],
                "training_file": config["training_file"],
                "training_file_oai_id": config["training_file_oai_id"],
                "test_file": config["test_file"] if "test_file" in config else None,
                "test_file_oai_id": config["test_file_oai_id"] if "test_file_oai_id" in config else None,
                "hyperparameters": config["hyperparameters"],
                "ft_job_id": response.id,
            }

        except Exception as e:
            logger.exception(f"Error running experiment with config {config}: {str(e)}")
    
    output_path = Path(__file__).parent / "_experiments.json"
    with open(output_path, "w") as f:
        json.dump(experiments, f, indent=4)
    logger.info(f"Generated {len(experiments)} experiments.")
    logger.info(f"Experiments saved to {output_path}")
    
    return experiments

