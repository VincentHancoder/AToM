import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import Tuple
from functools import partial
from transformers import logging as transformers_logging


logger = transformers_logging.get_logger("transformers")

def _write_dataset_ids(dataset: Dataset, root_path: str, is_train: bool = True):
    file_name = "train.txt" if is_train else "eval.txt"

    with open(os.path.join(root_path, file_name), 'w') as f:
        for item in dataset:
            f.write("%s\n" % item["id"])
            
            
def _check_ids_match(train_dataset, root_path: str, eval_dataset):
    train_file_name = "train.txt"
    eval_file_name = "eval.txt"
    # Read IDs from txt files
    with open(os.path.join(root_path, train_file_name), 'r') as f:
        train_ids = f.read().splitlines()
    with open(os.path.join(root_path, eval_file_name), 'r') as f:
        eval_ids = f.read().splitlines()

    # Extract IDs from datasets
    train_dataset_ids = [str(item['id']) for item in train_dataset]
    eval_dataset_ids = [str(item['id']) for item in eval_dataset]

    # Check if IDs match
    return train_ids == train_dataset_ids and eval_ids == eval_dataset_ids

def _preprocess_dpo(
    examples, 
    codebook_size: int, 
    use_label_smoothing,
    base_dir,
):
    """
        Examples (assumes batched):
        {
            "ID": [1, ...], 
            "prompt": ["text", ...],
            "sample_1": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 1234
                },
                ...
            ]
                
            "sample_2": [
                {
                    "gif": "path_to_gif", 
                    "array": "path_to_npy",
                    "from": "motionGPT",
                    "seed": 5678
                },
                ...
            ]
            "chosen": [
                [{ 
                    "choice": "sample_i", 
                    "degree of preference": "null",
                    "user": "Matt"
                }],	
                ...
            ] 
        }
    """
    
    new_examples = {
        "prompt": [],
        "chosen": [],
        "chosen_len": [],
        "rejected": [],
        "rejected_len": [],
        "label_smoothing": []
    }
        
    def load_pairwise_token(
        prompt: str,
        sample_1: dict, 
        sample_2: dict, 
        choice: str
    ) -> Tuple[str, str]:
        def load_motion_token_and_len(sample):
            motion_token_path = sample["token"]
            motion_token_path = os.path.join(base_dir, motion_token_path)
            motion_token_length_path = os.path.splitext(motion_token_path)[0] + "_len.npy"
            return np.load(motion_token_path), np.load(motion_token_length_path)
        # motion_token_to_string = lambda motion_token: f'<motion_id_{codebook_size}>' + ''.join([f'<motion_id_{int(i)}>' for i in motion_token.tolist()]) + f'<motion_id_{codebook_size + 1}>'

        sample_1_token_and_len = load_motion_token_and_len(sample_1)
        sample_2_token_and_len = load_motion_token_and_len(sample_2)

        # import ipdb
        # ipdb.set_trace()
        # pref_motion_string = motion_token_to_string(sample_1_token if choice == "sample_1" else sample_2_token)
        # dispref_motion_string = motion_token_to_string(sample_2_token if choice == "sample_1" else sample_1_token)
        
        # return (
        #     prompt,
        #     pref_motion_string,
        #     dispref_motion_string
        # )

        pref_motion_token_and_len = sample_1_token_and_len if choice == "sample_1" else sample_2_token_and_len
        dispref_motion_token_and_len = sample_2_token_and_len if choice == "sample_1" else sample_1_token_and_len

        return (
            prompt,
            pref_motion_token_and_len,
            dispref_motion_token_and_len
        )
    
    for prompt, sample_1, sample_2, choice in zip(examples["prompt"], examples["sample_1"], examples["sample_2"], examples["chosen"]):
        prompt, chosen, rejected = load_pairwise_token(prompt, sample_1, sample_2, choice[-1]['choice'])

        new_examples["prompt"].append(prompt)
        new_examples["chosen"].append(chosen[0])
        new_examples["rejected"].append(rejected[0])

        new_examples["chosen_len"].append(chosen[1])
        new_examples["rejected_len"].append(rejected[1])

        if use_label_smoothing:
            if choice[-1]["degree of preference"] == "Much better":
                label_smoothing = 0.
            elif choice[-1]["degree of preference"] == "Better":
                label_smoothing = 0.2
            elif choice[-1]["degree of preference"] == "Slightly better":
                label_smoothing = 0.3
            elif choice[-1]["degree of preference"] == "Negligibly better/unsure":
                label_smoothing = 0.5
        else:
            label_smoothing = 0.
        new_examples["label_smoothing"].append(label_smoothing)
    return new_examples

def _load_dataset_from_jsonl(
    jsonl_path: str,
) -> Dataset:
    """Loads a jsonl huggingface dataset by turning it into a pandas dataframe.

    Args:
        jsonl_path (str): path to jsonl file to be converted into a dataset.

    Returns:
        Dataset: each data point is a preference.
    """
    data = list()
    with open(jsonl_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def build_dpo_dataset(
    use_label_smoothing: bool,
    jsonl_path: str,
    seed: int,
    root_path: str,
    codebook_size: int,
    sanity_check: bool = True, # checks if the dataset loaded the correct train eval samples
    percentage_used: float = 1.0,
    preference_type: str = "all",
    add_unsure: bool = False,
) -> Tuple[Dataset, Dataset]:
    dataset = _load_dataset_from_jsonl(jsonl_path)
    dataset = dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")

    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    # if use_skipped:
    #     eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
    
    if sanity_check:
        file_name_maybe_use_skipped = "train.txt"
        if not os.path.isfile(os.path.join(root_path, file_name_maybe_use_skipped)):
            _write_dataset_ids(train_dataset, root_path, is_train=True)
            _write_dataset_ids(eval_dataset, root_path, is_train=False)
        else:
            if _check_ids_match(train_dataset, root_path, eval_dataset):
                logger.info(f"Passed sanity check for dataset: {os.path.join(root_path, file_name_maybe_use_skipped)}")
            else:
                logger.info(f"Failed sanity check for dataset: {os.path.join(root_path, file_name_maybe_use_skipped)}")
    
    if not use_label_smoothing and not add_unsure:
        train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] != "Negligibly better/unsure")
        eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] != "Negligibly better/unsure")

    if preference_type != "all": # ["all", "much_better", "better", "slightly_better", "all_better"]
        if preference_type != "all_better":
            if "_" in preference_type:
                preference_type = preference_type.replace("_", " ")
            # capitalize first
            preference_type = preference_type[0].upper() + preference_type[1:]
            train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == preference_type)
            eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] == preference_type)
        else:
            train_dataset = train_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] in ["Much better", "Better"])
            eval_dataset = eval_dataset.filter(lambda x: x["chosen"][-1]["degree of preference"] in ["Much better", "Better"])
    # randomly take a percentage of the dataset
    train_dataset = Dataset.from_pandas(train_dataset.shuffle(seed).to_pandas().sample(frac=percentage_used))
    eval_dataset = Dataset.from_pandas(eval_dataset.shuffle(seed).to_pandas().sample(frac=percentage_used))
    
    preprocess_function_maybe_margin = partial(
        _preprocess_dpo, 
        codebook_size=codebook_size, 
        use_label_smoothing=use_label_smoothing,
        base_dir=root_path
        
    )
    train_dataset = train_dataset.map(
        preprocess_function_maybe_margin,
        batched=True,
        num_proc=1,
    )
    eval_dataset = eval_dataset.map(
        preprocess_function_maybe_margin,
        batched=True,
        num_proc=1,
    )
    
    return train_dataset, eval_dataset
    