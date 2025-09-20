from openai import OpenAI
import json
import logging
from .logging_config import setup_logger

logger = setup_logger(log_level=logging.DEBUG)

with open('secrets/openai_api_key.json', 'r') as f:
    credentials = json.load(f)

client = OpenAI(
  api_key=credentials.get('openai_api_key'),
)


def run_finetuning(training_file,
                   model,
                   ft_method_config=None,
                   ):

    logger.debug(f"Training file: {training_file}")
    logger.debug(f"Model: {model}")
    
    kwargs = {}
    if ft_method_config is not None:
        kwargs["method"] = ft_method_config
        logger.debug("Overriding default fine-tuning method config with custom config.")
        logger.debug(f"Fine-tuning method config: {ft_method_config}")

    response = client.fine_tuning.jobs.create(
        training_file=training_file,
        model=model,
        **kwargs
    )
    return response


def query_fted_model_chat_completion(model_id,
                     user_query,
                     system_role_content="You are a helpful assistant.",
                     temperature=0.0,
                     num_responses=1,
                     ):
    """
    Query the fine-tuned model with a user query and return the response.
    
    Args:
        model_id (str): The ID of the fine-tuned model.
        user_query (str): The user's query.
        temperature (float): Sampling temperature. Must be between 0 and 2.
                             Higher values like 0.8 will make the response more random and creative, while
                             lower values like make it more deterministic.
        num_responses (int): The number of responses to generate. Default is 1.
    
    Returns:
        list: A list of responses from the model. The size of the list is equal to num_responses.

    """
    completion = client.chat.completions.create(
        model=model_id,
        n=num_responses,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_role_content},
            {"role": "user", "content": user_query}],
            # response_format={"type": "json_object"}
            )
    responses = []
    for i in range(num_responses):
        responses.append(completion.choices[i].message.content)
    return responses


def query_fted_model_responses(model_id,
                             user_query,
                             system_role_content="You are a helpful assistant.",
                             temperature=0.0,
                             num_responses=1,
                             ):
    """
    Query the fine-tuned model with a user query using the responses API and return the response.
    
    Args:
        model_id (str): The ID of the fine-tuned model.
        user_query (str): The user's query.
        system_role_content (str): The system role content for the prompt.
        temperature (float): Sampling temperature. Must be between 0 and 2.
                             Higher values like 0.8 will make the response more random and creative, while
                             lower values like make it more deterministic.
        num_responses (int): The number of responses to generate. Default is 1.
    
    Returns:
        list: A list of responses from the model. The size of the list is equal to num_responses.
    """
    response = client.responses.create(
        model=model_id,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_role_content},
            {"role": "user", "content": user_query}],
            )
    return response.output_text


if __name__ == "__main__":

    # training_file_id = 'file-G8tstQfCKpgCcE8mzzoxrC'
    # validation_file = 'file-XnGVfEupYMWCgYfDUM3cvf'
    # suffix = 'views_18_4_25_on_epochs_4'
    # model = 'gpt-4o-2024-08-06'
    
    # custom_method_config = {
    #     "type": "supervised",
    #     "supervised": {
    #         "hyperparameters": {
    #             "batch_size": "auto",
    #             "learning_rate_multiplier": "auto",
    #             "n_epochs": 4,
    #         }
    #     },
    # }


    # Query Classification
    training_file_id = 'file-W5cHvk3RJNv3VfTVcqf4zg'
    validation_file = 'file-7YSFtRgrAS7iTUWXcFiT3Q'
    suffix = 'queries_dsv_23_4_25' # dsv for dataset version
    model = 'gpt-4.1-mini-2025-04-14'

    # custom_method_config = {
    #     "type": "supervised",
    #     "supervised": {
    #         "hyperparameters": {
    #             "batch_size": "auto",
    #             "learning_rate_multiplier": "auto",
    #             "n_epochs": 4,
    #         }
    #     },
    # }

    logger.info("Starting fine-tuning job...")
    res = run_finetuning(training_file=training_file_id,
                model=model,
                suffix=suffix,
                )
    logger.info(f"Response from the API: {res}")