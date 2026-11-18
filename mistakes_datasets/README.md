 python -m mistakes_datasets.corrections_raw --data_dirname data\corrections --raw_datasets hitom:data\\hitom\\0000.parquet:answer:bot_answer:prompt --raw_datasets "lifechoices:data\\lifechoices\\0000.parquet:Character's Choice:answer:prompt" --raw_datasets mistaks:data\\mistakes\\0000.parquet:CorrectAnswer:TargetOption:raw_prompt

python -m mistakes_datasets.lifechoices_dataset --data_dirname data\lifechoices

python -m mistakes_datasets.hitom_dataset --data_dirname data\hitom


 python -m mistakes_datasets.corrections_raw --data_dirname data\corrections --raw_datasets hitom:data\\hitom\\0000.parquet:answer:bot_answer:prompt --raw_datasets "lifechoices:data\\lifechoices\\0000.parquet:Character's Choice:answer:prompt" --raw_datasets mistaks:data\\mistakes\\0000.parquet:CorrectAnswer:TargetOption:raw_prompt --llm_model ollama_chat/gemma4:12b --llm_api_base http://localhost:11434

 python -m mistakes_datasets.create_or_update_prompts --mlflow_prompt_name corrections-prompt-template --commit_message "init" --prompt_file_path mistakes_datasets\prompts\corrections.md