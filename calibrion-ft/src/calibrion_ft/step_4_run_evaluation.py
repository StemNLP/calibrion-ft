from evaluation.core import run_evaluators
from evaluation.registry import get_evaluator_registry
from evaluation.evaluation_utilities import format_eval_results, extract_code
import wandb
from pathlib import Path
import json
import logging
import pandas as pd
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.INFO)

def evaluate_ft_model(model_results: list, evaluator_registry) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all responses for a single model and return aggregated results.
    
    Args:
        model_results: List of dictionaries containing evaluation data for one model
        evaluator_registry: Registry of evaluators to use
    
    Returns:
        tuple: (detailed_df, summary_df) - Aggregated evaluation results
        Returns (None, None) if evaluation fails or input is empty
    """
    if not model_results:
        logger.warning("Empty model results provided")
        return None, None

    agg_error_details_df = None
    
    for eval_run in model_results:
        expected_response = eval_run["expected_response"]
        generated_response = eval_run["generated_response"]
        datapoint_id = eval_run["datapoint_id"]
        
        try:
            expected_html = extract_code(expected_response, "html_code")
            generated_html = extract_code(generated_response, "html_code")
            expected_js = extract_code(expected_response, "js_code")
            generated_js = extract_code(generated_response, "js_code")
        except Exception:
            logger.error(f"Code extraction failed for datapoint {datapoint_id}")
            continue

        # TODO Clarify translations handling
        if generated_html:
            translation_code = generated_html
        elif generated_js:
            translation_code = generated_js
        eval_input = {
            "html_code": generated_html,
            "js_code": generated_js,
            "js_codes": [generated_js],
            "html_codes": [generated_html],
            "translations": translation_code,
        }
        logger.debug(f"Running evaluators for datapoint {datapoint_id} with input: {eval_input}")
        try:
            result = run_evaluators(evaluator_registry,
                              eval_input,
                              skip_evaluators=["semantic_similarity_evaluator", "multi_template_guidelines"])
        except Exception as e:
            logger.error(f"Error running evaluators for datapoint {datapoint_id}: {str(e)}")
            continue
        
        details_df, _ = format_eval_results(result, method='pandas')
        
        if agg_error_details_df is None:
            agg_error_details_df = details_df.copy()
        else:
            agg_error_details_df['Count'] += details_df['Count']        

    return agg_error_details_df


def evaluate_all_ft_models(wandb_project: str) -> None:
    """
    Evaluate all model results from the JSON file.
    
    Args:
        wandb_project: Name of the W&B project to log results
    """
    eval_run_results = Path(__file__).parent / "_ft_models_eval_runs.json"
    experiments_path = Path(__file__).parent / "_experiments.json"
    
    logger.info(f"Starting evaluation of results from {eval_run_results}")
    evaluator_registry = get_evaluator_registry()

    with open(experiments_path, 'r') as f:
        experiments_config = json.load(f)
    
    with open(eval_run_results, 'r') as f:
        ft_models_eval_runs = json.load(f)
    
    for ft_model_id, content in ft_models_eval_runs.items():
        experiment_config = None
        for exp in experiments_config.values():
            if exp.get('ft_model_id') == ft_model_id:
                experiment_config = exp
                break
        
        if not experiment_config:
            logger.warning(f"No experiment config found for model {ft_model_id}")
            continue

        wandb.init(
            project=wandb_project,
            name=f"evaluation_{ft_model_id}",
            config={
                "model": experiment_config["model"],
                "training_file": experiment_config["training_file"],
                "hyperparameters": experiment_config["hyperparameters"],
                "ft_job_id": experiment_config["ft_job_id"],
                "ft_model_id": ft_model_id
            },
            tags=["model_evaluation"],
            reinit=True
        )
        
        logger.info(f"Evaluating results for model {ft_model_id}")
        details_df = evaluate_ft_model(content, evaluator_registry)
        if details_df is not None:
            logger.info(f"\nModel {ft_model_id} Detailed Results:\n{details_df}\n")
            metrics = {}
            for _, row in details_df.iterrows():
                category = row['Category'].lower().replace(' ', '_')
                count = row['Count']
                metrics[f"errors/{category}"] = count
            
            metrics["errors/total"] = details_df['Count'].sum()

            wandb.log(metrics)
            
        wandb.finish()