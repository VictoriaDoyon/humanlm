import mlflow
from openai import OpenAI
from mlflow.genai.scorers import Correctness, Safety
import pandas as pd
from mlflow.genai.datasets import create_dataset, get_dataset, EvaluationDataset
from polars import DataFrame as pldf
from datasets import Dataset
from mlflow.data import polars_dataset, pandas_dataset, huggingface_dataset, get_source
from datetime import datetime
import tempfile
import os

# class LocalEvaluationDataset():
#     def __init__(self, x, y):
#         self.list = [{
#             "inputs": input,
#             "expectations": expectation
#         } for input, expectation in zip(x, y)]
#         self.dataframe = pd.DataFrame(self.list)
#         self.mlflow_dataset = None

#     def __init__(self, df: pldf):
#         self.dataframe = df
#         self.list = df.to_dicts()
#         self.mlflow_dataset = None

#     def __init__(self, ds: Dataset):
#         self.dataframe = ds.to_polars()
#         self.list = ds.to_list()
#         self.mlflow_dataset = None

class CustomVersionedDataset():
    def _init_dataset(self, base_name: str, version: str, ds: Dataset | pldf | pd.DataFrame):
        init_dataset = None
        pd_df = None
        ds_type = type(ds)
        if ds_type == pldf:
            init_dataset = polars_dataset.from_polars(
                ds,
                source=f"{base_name}-v{version}.json",
                name=f"{base_name}-v{version}",
            )
            pd_df = init_dataset.df.to_pandas()
        elif ds_type == pd.DataFrame:
            init_dataset = pandas_dataset.from_pandas(
                ds,
                source=f"{base_name}-v{version}.json",
                name=f"{base_name}-v{version}",
            )
            pd_df = init_dataset.df
        elif ds_type == Dataset:
            init_dataset = huggingface_dataset.from_huggingface(
                ds,
                source=f"{base_name}-v{version}.json",
                name=f"{base_name}-v{version}",
            )
            pd_df = init_dataset.ds.to_pandas()
        else:
            raise ValueError("ds must be Dataset, polar dataframe, or pandas dataframe")
        return init_dataset, pd_df
    
    def __init__(self, base_name: str, version: str, ds: Dataset | pldf | pd.DataFrame, load_only: bool = False):
        self.dataset, self._pandas_dataframe = self._init_dataset(base_name, version, ds)
        self.base_name = base_name
        self.version = version
        if not load_only:
            self._create_or_update_version()

    def _create_or_update_version(self, new_version: str = None, new_dataset: Dataset | pldf | pd.DataFrame | None = None):
        if new_version and (new_dataset is not None) and (new_version != self.version):
            print(f"Updating dataset version from {self.version} to {new_version}")
            self.version = new_version
            self.dataset, self._pandas_dataframe = self._init_dataset(new_dataset)
        with mlflow.start_run(run_name=f"{self.base_name}-v{self.version}-{datetime.now()}"):
            mlflow.log_input(self.dataset, context="versioning",tags={"dataset_name": self.base_name, "dataset_version": self.version})

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_json_path = os.path.join(tmp_dir, f"{self.base_name}-v{self.version}.json")
                self._pandas_dataframe.to_json(local_json_path, orient="records", indent=2)
                mlflow.log_artifact(local_json_path)

class MLFlowService():
    def __init__(self, experiment_name, server_url = None):
        # Enable auto-tracing for OpenAI (works with Ollama)
        mlflow.langchain.autolog()
        if server_url:
            # Optional: Set a tracking URI
            mlflow.set_tracking_uri(server_url)
            self.client = OpenAI(
                base_url=server_url,
            )

        self.experiment = mlflow.set_experiment(experiment_name)

    def create_or_update_versioned_dataset(self, name: str, version: str, ds: Dataset | pldf | pd.DataFrame):
        versioned_ds = self.try_get_versioned_dataset(name, version)
        if versioned_ds:
            versioned_ds._create_or_update_version(version, ds)
        else:
            versioned_ds = CustomVersionedDataset(name, version, ds)

        return versioned_ds

    def try_get_versioned_dataset(self, name: str, version: str):
        target = f"{name}-v{version}"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"dataset.name = '{target}'",
            output_format="list"
        )
        if len(runs) == 0:
            return None
        latest = runs[0]
        local_file = mlflow.artifacts.download_artifacts(
            run_id=latest.info.run_id,
            artifact_path=f"{target}.json",
        )
        df = pd.read_json(local_file)
        return CustomVersionedDataset(name, version, df, load_only=True)


    # def run_evaluation(self, dataset: LocalEvaluationDataset, agent, scorers):
    #     '''
    #         Run model evaluation based on given dataset
    #         dataset: EvaluationDataset object
    #         agent: model or function under test
    #         scorers: list of scorer objects to be used for evaluation. E.G. [Correctness(), Safety()]
    #     '''

    #     # Run evaluation
    #     results = mlflow.genai.evaluate(
    #         data=dataset.dataframe,
    #         predict_fn=agent,
    #         scorers=scorers,
    #     )

    # def track_create_dataset(self, dataset_name: str, tags_dict, dataset: LocalEvaluationDataset):
    #     '''
    #         Create or update a dataset inside MLFlow
    #         Args:
    #             dataset_name: name of the dataset. To update a dataset, use the same name
    #             experiment_id: dataset version
    #             tags_dict: a dictionary type with the tags you'd like to use in this dataset version
    #             dataset: LocalEvaluationDataset object
    #     '''
    #     # Create dataset
    #     dataset_mlflow = create_dataset(
    #         name=dataset_name,
    #         experiment_id=[self.experiment.experiment_id],
    #         tags=tags_dict
    #     )

    #     # Add records from pandas DataFrame
    #     dataset.mlflow_dataset = dataset_mlflow.merge_records(dataset.dataframe.to_pandas())

    # def track_get_dataset(self, dataset_name: str, ):

    #     get_dataset(
    #         dataset_name
    #     )

    def track_prompt(self, prompt_name, prompt_template, commit_message):
        '''
            Create or update prompt in MLFlow system. Use same name to update prompt
        '''
        mlflow.genai.register_prompt(
            name=prompt_name,
            template=prompt_template,
            commit_message=commit_message
        )

# #     def track_prompt_from_md_file(self, prompt_name, prompt_template_path, commit_message):
# #         '''
# #             Create or update prompt in MLFlow system. Use same name to update prompt
# #         '''
# #         prompt_template = #read file
# #         mlflow.genai.register_prompt(
# #             name=prompt_name,
# #             template=prompt_template,
# #             commit_message=commit_message
# #         )

    def get_latest_prompt_by_name(self, prompt_template_name):
        return mlflow.genai.load_prompt(f"prompts:/{prompt_template_name}@latest")
