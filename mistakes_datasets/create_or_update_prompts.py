import argparse
from mlflow_service import MLFlowService

VERSION = "0.0.1"

class CreateOrUpdatePrompt():
    def __init__(self, mlflow_prompt_name, commit_message, prompt_file_path):
        self.prompt_name = mlflow_prompt_name
        self.commit = commit_message
        self.file = prompt_file_path

    def create_prompt(self, mlflow_service: MLFlowService):
        ver = mlflow_service.track_prompt_from_md_file(self.prompt_name, self.file, self.commit)
        return f"Prompt {self.prompt_name} updated to version {ver}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or update custom prompts")
    parser.add_argument(
        "--mlflow_prompt_name",
        type=str,
        default="data",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="data",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default="data",
        help="Directory to save the dataset",
    )
    args = parser.parse_args()
    
    mlflow_service = MLFlowService(
        f"create_or_update_prompts-v{VERSION}"
    )

    task = CreateOrUpdatePrompt(
        mlflow_prompt_name=args.mlflow_prompt_name,
        commit_message=args.commit_message,
        prompt_file_path=args.prompt_file_path
    )

    task.create_prompt(mlflow_service=mlflow_service)
