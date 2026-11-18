import argparse
import asyncio
from datetime import datetime
from datasets import Dataset
from .utils import memory_usage
from mlflow_service import MLFlowService
import os
import polars as pl
import copy
import time
import litellm
from .extract_json import extract_json
import mlflow

DATASET_NAME = "Corrections"
DATASET_VERSION = "0.0.3"

CORRECTIONS_FEEDBACK_TEMPLATE = mlflow.genai.load_prompt("prompts:/corrections-prompt-template@latest")

SCHEMA = pl.Schema({
    "answer": pl.String,
    "correction_record": pl.Struct({
        "is_correct": pl.Boolean,
        "correct_answer": pl.String,
        "correction": pl.String,
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
            llm_kwargs: dict,
            max_concurrent_users = 1,
            num_retries = 5,
            push_to_hub: str = None,
            save_local: str = None, 
        ):
            self.raw_tasks_datasets = raw_tasks_datasets
            self.push_to_hub = push_to_hub
            self.save_local = save_local
            self.sem = asyncio.Semaphore(max_concurrent_users)
            self.num_retries = num_retries
            self.llm_kwargs = llm_kwargs

    async def generate_correction(self, data: dict) -> tuple[str, dict]: ## TODO: ADD PROGRESS BAR
            llm_kwargs = copy.deepcopy(self.llm_kwargs)
            async with self.sem:
                for i in range(self.num_retries):
                    try:
                        prompt = CORRECTIONS_FEEDBACK_TEMPLATE.format(input=data["original_input"], answer=data["bot_answer"], is_correct=str(data["is_correct"]), correct_answer=data["correct_answer"])
                        cost_info = {"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
                        response_obj = await litellm.acompletion(
                            messages=[{"role": "user", "content": prompt}],
                            **llm_kwargs
                        )
                        response = response_obj.choices[0].message.content
    
                        # Extract cost info from litellm response
                        if hasattr(response_obj, 'usage') and response_obj.usage:
                            cost_info["prompt_tokens"] = getattr(response_obj.usage, 'prompt_tokens', 0)
                            cost_info["completion_tokens"] = getattr(response_obj.usage, 'completion_tokens', 0)
                            cost_info["total_tokens"] = getattr(response_obj.usage, 'total_tokens', 0)
    
                        # Get cost from litellm
                        try:
                            cost_info["cost_usd"] = float(litellm.completion_cost(response_obj))
                        except Exception:
                            # Fallback: cost will be 0 if not available
                            cost_info["cost_usd"] = 0.0
    
                        correction_text = response.strip() ##TODO if empty string consider failure
                        # print(correction_text)
                        # # correction = extract_json(correction_text)
    
                        print(f"Generated correction on attempt {i+1}: \n{correction_text}")
                        return correction_text, cost_info
    
                    except Exception as e:
                        if i < self.num_retries - 1:
                            if "vllm" in str(e).lower():
                                print(f"vLLM exception; early stopping. E: {e}")
                                raise e
                            await asyncio.sleep(60)
                            llm_kwargs['temperature'] = max(llm_kwargs.get('temperature', 0), 0.2)
                            print(f"Error generating correction: {e} Retrying...")
                        else:
                            raise RuntimeError(f"Error generating correction: {e} Failed after {self.num_retries} retries.") ##Probably will want to return instead of raise for the workspace
    
                # If we get here, all retries failed
                raise RuntimeError (f"Failed to generate correction after {self.num_retries} retries") ##Probably will want to return instead of raise for the workspace
    
    async def create_raw_dataset(self, mlflow_service: MLFlowService):
        # I don't want to fight with pull from hub so just doing local files for now
        if len(self.raw_tasks_datasets) == 0:
            raise ValueError("You must include at least one raw task dataset")

        corr_list = []
        for raw_tasks_dataset in self.raw_tasks_datasets: ## TODO: ADD PROGRESS BAR
            ds = pl.read_parquet(raw_tasks_dataset.parquet_file_path)
            answer_column = ds.get_column(raw_tasks_dataset.answer_field).alias("answer")
            correct_answer_column = ds.get_column(raw_tasks_dataset.correct_answer_field).alias("correct_answer_column")
            is_correct_column = (answer_column == correct_answer_column).alias("is_correct_column")
            input_column = ds.get_column(raw_tasks_dataset.input_field).alias("input_column")
            inputs = pl.DataFrame(
                [
                    input_column.alias("original_input"),
                    answer_column.alias("bot_answer"),
                    is_correct_column.alias("is_correct"),
                    correct_answer_column.alias("correct_answer"),
                ]
            )

            results = []
            batch_size = 100
            for batch in inputs.iter_slices(n_rows=batch_size):
                batch_results = await asyncio.gather(
                    *(self.generate_correction(row) for row in batch.iter_rows(named=True))
                )
                results.extend(batch_results)
            correction_column = pl.Series(
                                "correction",
                                [correction for correction, _ in results],
                                dtype=pl.String,
                            )
            corr_list.append(answer_column) 
            corr_list.append(input_column) 
            corr_list.append(
                pl.DataFrame([is_correct_column, correct_answer_column, correction_column]).select(
                    pl.struct(
                        is_correct="is_correct_column", 
                        correct_answer="correct_answer_column",
                        correction="correction_column"
                    ).alias("correction_record")
                ).to_series()
            )
            corr_list.append(
                            pl.Series("cost_usd", [cost["cost_usd"] for _, cost in results], dtype=pl.Float64),
                            pl.Series("prompt_tokens", [cost["prompt_tokens"] for _, cost in results], dtype=pl.Int64),
                            pl.Series("completion_tokens", [cost["completion_tokens"] for _, cost in results], dtype=pl.Int64),
                            pl.Series("total_tokens", [cost["total_tokens"] for _, cost in results], dtype=pl.Int64),
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
        "--llm_model",
        type=str,
        default="hosted_vllm/Qwen/Qwen3.8-27B",
        help="LLM model to use for correction generation",
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.0,
        help="LLM temperature to use for correction generation",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=4096,
        help="LLM max tokens to use for correction generation",
    )
    parser.add_argument(
        "--llm_api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="LLM max tokens to use for correction generation",
    )
    parser.add_argument(
        "--max_concurrent_threads",
        type=int,
        default=1,
        help="Number of concurrent LLM calls for correction generation",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=3,
        help="Number of retries before giving up on a row's correction generation",
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
        max_concurrent_users=args.max_concurrent_threads,
        num_retries=args.num_retries,
        push_to_hub=args.push_to_hub,
        save_local=args.data_dirname,
        llm_kwargs={
            "model": args.llm_model,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "api_base": args.llm_api_base
        }
    )

    asyncio.run(dataset.create_raw_dataset(mlflow_service=mlflow_service))
