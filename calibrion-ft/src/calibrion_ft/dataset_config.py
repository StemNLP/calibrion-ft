from pathlib import Path
import yaml
from typing import Dict, Optional

def get_dataset_config(version: str) -> Dict:
    """
    Get dataset configuration for a specific version from versions.yaml.
    
    Args:
        version (str): Dataset version (e.g., '1.1.small')
        
    Returns:
        dict: Dataset configuration including file paths and cloud IDs
    """
    versions_file = Path(__file__).parents[2] / "training_datasets/views/versions.yaml"
    
    with open(versions_file, 'r') as f:
        versions_data = yaml.safe_load(f)
    
    for dataset in versions_data['datasets']:
        if dataset['version'] == version:
            return dataset
            
    raise ValueError(f"Dataset version {version} not found in versions.yaml")

def get_dataset_files(version: str) -> tuple[str, Optional[str], str, Optional[str]]:
    """
    Get training and test file information for a dataset version.
    
    Args:
        version (str): Dataset version
        
    Returns:
        tuple: (training_file, test_file, training_file_oai_id, test_file_oai_id)
    """
    dataset = get_dataset_config(version)
    folder_name = dataset['folder'].replace('.', '_')
    base_path = Path(__file__).parents[2] / "training_datasets/views" / folder_name
    
    train_data = dataset['splits']['training']
    test_data = dataset['splits'].get('test', {})
    
    def convert_filename(filename: str) -> str:
        """Convert only the filename part, preserving extension"""
        name, ext = filename.rsplit('.', 1)
        return f"{name.replace('.', '_')}.{ext}"
    
    train_file = convert_filename(train_data['file'])
    test_file = convert_filename(test_data['file']) if test_data else None
    
    return (
        str(base_path / train_file),
        str(base_path / test_file) if test_file else None,
        train_data.get('cloud', {}).get('file_id'),
        test_data.get('cloud', {}).get('file_id') if test_data else None
    )
