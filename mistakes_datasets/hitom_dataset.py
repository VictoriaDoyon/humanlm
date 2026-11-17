"""
Original dataset: https://huggingface.co/datasets/Hi-ToM/Hi-ToM_Dataset
Original paper: https://arxiv.org/html/2404.12138v2 (Character is Destiny: Can Role-Playing Language Agents Make Persona-Driven Decisions?)
"""

import argparse
import asyncio
from datetime import datetime

from datasets import Dataset
import argparse
from .utils import memory_usage
from datetime import datetime
from mlflow_service import MLFlowService
import os
import polars as pl

DATASET_NAME = "Hi-ToM/Hi-ToM_Dataset"
DATASET_VERSION = "1.0.0"

class HiToMDataset:
    def __init__(
        self,
        push_to_hub: str = None,
        save_local: str = None,
    ):
        self.push_to_hub = push_to_hub
        self.save_local = save_local

    async def create_raw_dataset(self, mlflow_service: MLFlowService):
        hitom = pl.read_parquet(f"hf://datasets/{DATASET_NAME}@~parquet/default/train/*.parquet")
        hitom.with_columns(
            {
                "bot_answer": [],
                "prompt": [],
            }
        )
        print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
        if self.save_local or self.push_to_hub:
            mlflow_service.create_or_update_versioned_dataset(
                "hitom",
                DATASET_VERSION,
                hitom
            )
        if self.save_local:
            Dataset.from_polars(hitom).to_parquet(os.path.join(self.save_local, "0000.parquet"))
        if self.push_to_hub:
            Dataset.from_polars(hitom).push_to_hub(self.push_to_hub, config_name="hitom", split="train", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hitom dataset from OdySim - Formatting")
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
        f"hitom-v{DATASET_VERSION}",
    )

    dataset = HiToMDataset(
        push_to_hub=args.push_to_hub,
        save_local=args.data_dirname
    )

    asyncio.run(dataset.create_raw_dataset(mlflow_service=mlflow_service))
