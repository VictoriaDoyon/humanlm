"""
Original dataset: https://github.com/airaer1998/LifeChoice/tree/main
Original paper: https://arxiv.org/html/2404.12138v2 (Character is Destiny: Can Role-Playing Language Agents Make Persona-Driven Decisions?)
"""

import argparse
import asyncio
from datetime import datetime

from datasets import Dataset
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
DATASET_VERSION = "1.0.1"

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

    def create_prompt(self, character_data: dict) -> str:
        """Create structured prompt source: https://github.com/airaer1998/LifeChoice/blob/main/test_llms.py"""
        prompt = f"""Please play the role of {character_data['character_name']} based on the Profile and make your life choice under the Scenario regarding Question. Return the option letter (A, B, C, or D) that your character should most appropriately choose in the current scenario. The Profile consists of Description and Memory, where Description is an overall description of the character, and Memory consists of specific events the character has experienced.

# Inputs:
1. Profile:
1.1. Description
{character_data['character_name']}

1.2. Memory
{character_data['input_text']}

2. Scenario:
{character_data['Multiple Choice Question']['Scenario']}

3. Question:
{character_data['Multiple Choice Question']['Question']}

4. Options:
A. {character_data['Multiple Choice Question']['Options'][0]}
B. {character_data['Multiple Choice Question']['Options'][1]}
C. {character_data['Multiple Choice Question']['Options'][2]}
D. {character_data['Multiple Choice Question']['Options'][3]}

# Outputs:
Your choice(A, B, C, or D):"""
        return prompt

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
        lifechoices = lifechoices.filter(~pl.col("Choice's Location").is_in(["N/A", "Not applicable"]))
        mcq = lifechoices.get_column("Multiple Choice Question")
        correct_answer = mcq.struct.field("Correct Answer")
        incorrect_answer = (
            mcq.struct.field("Options").list.eval(pl.element().str.slice(0, 1))
            .list.set_difference(correct_answer.str.split(""))
            .list.sample(1).list.first().alias("answer")
        )

        lifechoices = lifechoices.with_columns(
            pl.concat_list(correct_answer, incorrect_answer)
                    .list.sample(n=1, seed=42)
                    .list.first().alias("answer"),
            pl.struct(
                    "character_name",
                    "input_text",
                    "Multiple Choice Question",
                )
                .map_elements(
                    self.create_prompt,
                    return_dtype=pl.String,
                ).alias("prompt")
        )

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
