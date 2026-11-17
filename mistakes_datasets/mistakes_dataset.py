"""
Dataset: https://huggingface.co/datasets/cmu-lti/osim-post-training/viewer/mistakes_train
Original paper: https://arxiv.org/html/2510.11502v1 (Learning To Make MISTAKEs: Modeling Incorrect Student Thinking And Key Errors)
"""

import argparse
import asyncio
from datetime import datetime

from datasets import Dataset, Features, Value, load_dataset
import argparse
from datetime import datetime
from .utils import memory_usage
from mlflow_service import MLFlowService
import os

DATASET_NAME = "cmu-lti/osim-post-training"
DATASET_VERSION = "1.0.3"

FEATURES = Features(
    {
        "AnswerAText": Value(dtype="string", id=None),
        "AnswerBText": Value(dtype="string", id=None),
        "AnswerCText": Value(dtype="string", id=None),
        "AnswerDText": Value(dtype="string", id=None),
        "CorrectAnswer":Value(dtype="string", id=None),
        "MisconceptionId": Value(dtype="string", id=None),
        "MisconceptionName": Value(dtype="string", id=None),
        "QuestionText": Value(dtype="string", id=None),
        "TargetAnswer": Value(dtype="string", id=None),
        "TargetOption": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "index": Value(dtype="int32", id=None),
        "raw_prompt": Value(dtype="string", id=None),
    }
)

class MistakesDataset:
    def __init__(
        self,
        push_to_hub: str = None,
        save_local: str = None
    ):
        self.push_to_hub = push_to_hub
        self.save_local = save_local

    async def create_raw_dataset(self, mlflow_service: MLFlowService):
        osim_post_training_train = load_dataset(DATASET_NAME, f"mistakes_train", trust_remote_code=True)["train"].to_polars()
        mistakes_dt = osim_post_training_train.get_column("extra_info").to_list()
        print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
        dataset = Dataset.from_list(mistakes_dt, features=FEATURES).remove_columns(["MisconceptionId", "id", "index"])
        if self.save_local or self.push_to_hub:
            mlflow_service.create_or_update_versioned_dataset(
                "mistakes",
                DATASET_VERSION,
                dataset
            )
        if self.save_local:
            dataset.to_parquet(os.path.join(self.save_local, "0000.parquet"))
        if self.push_to_hub:
            dataset.push_to_hub(self.push_to_hub, config_name="mistakes", split=f"train", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistakes dataset from OdySim - Formatting")
    parser.add_argument(
        "--data_dirname",
        type=str,
        default="data",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push dataset to Hugging Face Hub (optional)",
    )
    args = parser.parse_args()

    mlflow_service = MLFlowService(
        f"mistakes-v{DATASET_VERSION}",
    )

    dataset = MistakesDataset(
        push_to_hub=args.push_to_hub,
        save_local=args.data_dirname
    )

    asyncio.run(dataset.create_raw_dataset(mlflow_service=mlflow_service))
