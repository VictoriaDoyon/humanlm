"""
Original dataset: https://github.com/airaer1998/LifeChoice/tree/main
Original paper: https://arxiv.org/html/2404.12138v2 (Character is Destiny: Can Role-Playing Language Agents Make Persona-Driven Decisions?)
"""

import argparse
import asyncio
from datetime import datetime

from datasets import Dataset, Sequence, Features, Value, load_dataset
import argparse
from datetime import datetime
from .utils import memory_usage
from mlflow_service import MLFlowService
import os
import polars as pl
import requests
from io import StringIO
import json
import random


DATASET_NAME = "LifeChoices"
DATASET_URL = "https://raw.githubusercontent.com/airaer1998/LifeChoice/refs/heads/main/data/data.json"
DATASET_VERSION = "1.0.0"

FEATURES = pl.Schema({
    "Character's Choice": pl.String,
    "Choice's Location": pl.String,
    "Multiple Choice Question": pl.Struct({
        "Scenario": pl.String,
        "Question": pl.String,
        "Options": pl.List(pl.String),
        "Correct Answer": pl.String,
        "Motivations": pl.List(pl.String),
    }),
    "character_name": pl.String,
    "book": pl.String,
    "input_text": pl.String
})

class LifeChoicesDataset:
    def __init__(
        self,
        push_to_hub: str = None,
        save_local: str = None
    ):
        self.push_to_hub = push_to_hub
        self.save_local = save_local

    async def create_raw_dataset(self, mlflow_service: MLFlowService):
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        res_json = response.json()
        dataset = []
        for row in res_json:
            if isinstance(row, list):
                dataset.extend(row)
            else:
                dataset.append(row)
        lifechoices = pl.DataFrame(dataset, schema=FEATURES)
        mcq = lifechoices.select("Multiple Choice Question").to_series()
        incorrect_answer = mcq.struct.field("Options").list.filter(pl.element() != mcq.struct.field("Correct Answer")).list.sample(n=1, seed=50).alias("answer")
        
        # # corr_list.append(
        # #                 pl.DataFrame([is_correct_column, correct_answer_column]).select(
        # #                     pl.struct(
        # #                         is_correct="is_correct_column", 
        # #                         correct_answer="correct_answer_column"
        # #                     ).alias("correction_record"),
                            
        # #                 ).to_series()
        # #             )
        # # lifechoices.with_columns(
        # #     {
        # #         "answer": [],
        # #         "prompt": [],
        # #     }
        # # )
        print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
        if self.save_local or self.push_to_hub:
            mlflow_service.create_or_update_versioned_dataset(
                "lifechoices",
                DATASET_VERSION,
                lifechoices
            )
        if self.save_local:
            Dataset.from_polars(lifechoices).to_parquet(os.path.join(self.save_local, "0000.parquet"))
        if self.push_to_hub:
            Dataset.from_polars(lifechoices).push_to_hub(self.push_to_hub, config_name="lifechoices", split=f"train", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lifechoices dataset from OdySim - Formatting")
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
        f"lifechoices-v{DATASET_VERSION}",
    )

    dataset = LifeChoicesDataset(
        push_to_hub=args.push_to_hub,
        save_local=args.data_dirname
    )

    asyncio.run(dataset.create_raw_dataset(mlflow_service=mlflow_service))
