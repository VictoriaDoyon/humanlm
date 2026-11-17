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

DATASET_NAME = "Corrections"
DATASET_VERSION = "0.0.3"

SCHEMA = pl.Schema({
    "answer": pl.String,
    "correction_record": pl.Struct({
        "is_correct": pl.Boolean,
        "correct_answer": pl.String,
        # "correction": pl.String,
    }),
    "input": pl.String,
})

class RawTaskDataset:
    def __init__(self, 
        dataset_name: str,
        parquet_file_path: str, 
        correct_answer_field: str,
        answer_field: str,
        input_field: str
    ):
        self.name = dataset_name
        self.parquet_file_path = parquet_file_path
        self.correct_answer_field = correct_answer_field
        self.answer_field = answer_field
        self.input_field = input_field

        

class CorrectionsDataset:
    def __init__(
            self,
            raw_tasks_datasets: list[RawTaskDataset],
            push_to_hub: str = None,
            save_local: str = None
        ):
            self.raw_tasks_datasets = raw_tasks_datasets
            self.push_to_hub = push_to_hub
            self.save_local = save_local
    
    async def create_raw_dataset(self, mlflow_service: MLFlowService):
        # I don't want to fight with pull from hub so just doing local files for now
        if len(self.raw_tasks_datasets) == 0:
            raise ValueError("You must include at least one raw task dataset")

        corr_list = []
        for raw_tasks_dataset in self.raw_tasks_datasets:
            ds = pl.read_parquet(raw_tasks_dataset.parquet_file_path)
            answer_column = ds.get_column(raw_tasks_dataset.answer_field).alias("answer")
            correct_answer_column = ds.get_column(raw_tasks_dataset.correct_answer_field).alias("correct_answer_column")
            is_correct_column = (answer_column == correct_answer_column).alias("is_correct_column")
            input_column = ds.get_column(raw_tasks_dataset.input_field).alias("input_column")

            corr_list.append(answer_column) 
            corr_list.append(input_column) 
            corr_list.append(
                pl.DataFrame([is_correct_column, correct_answer_column]).select(
                    pl.struct(
                        is_correct="is_correct_column", 
                        correct_answer="correct_answer_column"
                    ).alias("correction_record")
                ).to_series()
            )
        corrections = pl.DataFrame(corr_list, schema=SCHEMA)
        print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
        if self.save_local or self.push_to_hub:
            mlflow_service.create_or_update_versioned_dataset(
                "corrections",
                DATASET_VERSION,
                corrections
            )
        if self.save_local:
            Dataset.from_polars(corrections).to_parquet(os.path.join(self.save_local, "0000.parquet"))
        if self.push_to_hub:
            Dataset.from_polars(corrections).push_to_hub(self.push_to_hub, config_name="mistakes", split=f"train", private=True)

def parse_raw_task_dataset_argument(arg: str):
    try:
        dataset_name, parquet_file_path, correct_answer_field, answer_field, input_field = arg.split(":")
        return RawTaskDataset(dataset_name, parquet_file_path, correct_answer_field, answer_field, input_field)
    except ValueError:
        raise argparse.ArgumentTypeError("Use : as a separator")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joining datasets to create corrections dataset")
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
    parser.add_argument(
        "--raw_datasets", 
        nargs="+", 
        type=parse_raw_task_dataset_argument, 
        required=True, 
        help="List of raw datasets used to build the corrections dataset."
    )
    args = parser.parse_args()

    mlflow_service = MLFlowService(
        f"corrections-v{DATASET_VERSION}"
    )

    dataset = CorrectionsDataset(
        raw_tasks_datasets=args.raw_datasets,
        push_to_hub=args.push_to_hub,
        save_local=args.data_dirname
    )

    asyncio.run(dataset.create_raw_dataset(mlflow_service=mlflow_service))
