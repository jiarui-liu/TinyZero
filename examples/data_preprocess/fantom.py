"""
Preprocess dataset for FANTOM
"""

import os
import json
import pandas as pd
import random
from datasets import Dataset
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse
from huggingface_hub import hf_hub_download, login
from sklearn.model_selection import train_test_split


def load_dataset(file_path: str) -> List[dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
        return data


def set_beliefQA_multiple_choices(qa):
    if qa["question_type"].endswith(":inaccessible"):
        option_a = qa["wrong_answer"]
        option_b = qa["correct_answer"]
    else:
        option_a = qa["wrong_answer"]
        option_b = qa["correct_answer"]

    answer_goes_last = random.choice([True, False])
    if answer_goes_last:
        choices = [option_a, option_b]
        answer = 1
    else:
        choices = [option_b, option_a]
        answer = 0

    # option letters iterate over the alphabet
    option_letters = [
        "(" + chr(x) + ")" for x in range(ord("a"), len(choices) + ord("a"))
    ]
    choices_text = ""
    for letter, option in zip(option_letters, choices):
        choices_text += "{} {}\n".format(letter, option)

    return choices_text, answer


def filter_dataset_list(data_lst):
    new_data_list = []
    for item in data_lst:
        # select only binary questions and binary choice questions
        if item["question_type"] not in [
            "tom:belief:accessible:multiple-choice",
            "tom:belief:inaccessible:multiple-choice",
            "tom:answerability:binary",
            "tom:info_accessibility:binary",
        ]:
            continue

        # for binary questions, replace "correct_answer": "no:long" with "no"
        if (
            item["question_type"]
            in ["tom:answerability:binary", "tom:info_accessibility:binary"]
            and item["correct_answer"] == "no:long"
        ):
            item["correct_answer"] = "no"

        # for binary choice questions, replace 0 with (a), and 1 with (b)
        if item["question_type"] in [
            "tom:belief:accessible:multiple-choice",
            "tom:belief:inaccessible:multiple-choice",
        ]:
            if item["correct_answer"] == 0:
                item["correct_answer"] = "(a)"
            elif item["correct_answer"] == 1:
                item["correct_answer"] = "(b)"

        new_data_list.append(item)
    return new_data_list


def process_dataset(data, aggregation_target="set", conversation_input_type="short"):
    fantom_df = pd.DataFrame(data)

    if aggregation_target == "conversation":
        assert (
            conversation_input_type == "full"
        ), "The input type should have been the full conversation. It doesn't make sense to aggregate the scores over the full conversation when the input is not the full conversation"

    fantom_df_to_run = fantom_df

    total_num_q = 0
    for idx, _set in fantom_df_to_run.iterrows():
        total_num_q += len(_set["beliefQAs"])
        total_num_q += len(_set["answerabilityQAs_binary"])
        total_num_q += len(_set["infoAccessibilityQAs_binary"])
        if _set["factQA"] is not None:
            total_num_q += 1
        if _set["answerabilityQA_list"] is not None:
            total_num_q += 1
        if _set["infoAccessibilityQA_list"] is not None:
            total_num_q += 1

    inputs = []
    qas = []
    for idx, _set in fantom_df_to_run.iterrows():
        if conversation_input_type == "short":
            context = _set["short_context"].strip()
        elif conversation_input_type == "full":
            context = _set["full_context"].strip()

        set_id = _set["set_id"]
        fact_q = _set["factQA"]["question"]
        fact_a = _set["factQA"]["correct_answer"]

        # Fact Question
        _set["factQA"]["context"] = context
        input_text = "{}\n\nQuestion: {}".format(context, fact_q)
        _set["factQA"]["input_text"] = input_text
        _set["factQA"]["set_id"] = set_id
        qas.append(_set["factQA"])
        inputs.append(input_text)

        for _belief_qa in _set["beliefQAs"]:
            # Belief Questions
            _belief_qa["context"] = context
            input_text = "{}\n\nQuestion: {}".format(context, _belief_qa["question"])
            _belief_qa["input_text"] = input_text
            _belief_qa["set_id"] = set_id
            qas.append(_belief_qa)
            inputs.append(input_text)

            # Multiple Choice Belief Questions
            _mc_belief_qa = {**_belief_qa}
            choices_text, answer = set_beliefQA_multiple_choices(_mc_belief_qa)
            mc_question = "{}\n{}\n\nChoose an answer from above:".format(
                _belief_qa["question"], choices_text.strip()
            )
            _mc_belief_qa["question"] = mc_question
            _mc_belief_qa["question_type"] = (
                _mc_belief_qa["question_type"] + ":multiple-choice"
            )
            _mc_belief_qa["choices_text"] = choices_text
            _mc_belief_qa["choices_list"] = choices_text.strip().split("\n")
            _mc_belief_qa["correct_answer"] = answer
            input_text = "{}\n\nQuestion: {}".format(context, mc_question)
            _mc_belief_qa["input_text"] = input_text
            qas.append(_mc_belief_qa)
            inputs.append(input_text)

        # Answerability List Questions
        _set["answerabilityQA_list"]["fact_question"] = fact_q
        _set["answerabilityQA_list"]["context"] = context
        input_text = "{}\n\nTarget: {}\nQuestion: {}".format(
            context, fact_q, _set["answerabilityQA_list"]["question"]
        )
        _set["answerabilityQA_list"]["input_text"] = input_text
        _set["answerabilityQA_list"]["set_id"] = set_id
        if (
            conversation_input_type == "full"
            and len(_set["answerabilityQA_list"]["wrong_answer"]) > 0
        ):
            _set["answerabilityQA_list"]["missed_info_accessibility"] = "inaccessible"
        qas.append(_set["answerabilityQA_list"])
        inputs.append(input_text)

        # Answerability Binary Questions
        if conversation_input_type == "full":
            missed_info_accessibility_for_full = _set["answerabilityQAs_binary"][0][
                "missed_info_accessibility"
            ]
            for _info_accessibility_qa in _set["answerabilityQAs_binary"]:
                if _info_accessibility_qa["correct_answer"] != "yes":
                    missed_info_accessibility_for_full = "inaccessible"

        for _answerability_qa in _set["answerabilityQAs_binary"]:
            _answerability_qa["fact_question"] = fact_q
            _answerability_qa["context"] = context
            input_text = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.".format(
                context, fact_q, _answerability_qa["question"]
            )
            _answerability_qa["input_text"] = input_text
            _answerability_qa["set_id"] = set_id
            if conversation_input_type == "full":
                _answerability_qa["missed_info_accessibility"] = (
                    missed_info_accessibility_for_full
                )
            qas.append(_answerability_qa)
            inputs.append(input_text)

        # Info Accessibility List Questions
        _set["infoAccessibilityQA_list"]["fact_question"] = fact_q
        _set["infoAccessibilityQA_list"]["fact_answer"] = fact_a
        _set["infoAccessibilityQA_list"]["context"] = context
        input_text = "{}\n\nInformation: {} {}\nQuestion: {}".format(
            context, fact_q, fact_a, _set["infoAccessibilityQA_list"]["question"]
        )
        _set["infoAccessibilityQA_list"]["input_text"] = input_text
        _set["infoAccessibilityQA_list"]["set_id"] = set_id
        if (
            conversation_input_type == "full"
            and len(_set["infoAccessibilityQA_list"]["wrong_answer"]) > 0
        ):
            _set["infoAccessibilityQA_list"][
                "missed_info_accessibility"
            ] = "inaccessible"
        qas.append(_set["infoAccessibilityQA_list"])
        inputs.append(input_text)

        # Info Accessibility Binary Questions
        if conversation_input_type == "full":
            missed_info_accessibility_for_full = _set["infoAccessibilityQAs_binary"][0][
                "missed_info_accessibility"
            ]
            for _info_accessibility_qa in _set["infoAccessibilityQAs_binary"]:
                if _info_accessibility_qa["correct_answer"] != "yes":
                    missed_info_accessibility_for_full = "inaccessible"

        for _info_accessibility_qa in _set["infoAccessibilityQAs_binary"]:
            _info_accessibility_qa["fact_question"] = fact_q
            _info_accessibility_qa["fact_answer"] = fact_a
            _info_accessibility_qa["context"] = context
            input_text = (
                "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.".format(
                    context, fact_q, fact_a, _info_accessibility_qa["question"]
                )
            )
            _info_accessibility_qa["input_text"] = input_text
            _info_accessibility_qa["set_id"] = set_id
            if conversation_input_type == "full":
                _info_accessibility_qa["missed_info_accessibility"] = (
                    missed_info_accessibility_for_full
                )
            qas.append(_info_accessibility_qa)
            inputs.append(input_text)

    qas = filter_dataset_list(qas)

    return qas


def split_dataset(train_dataset, test_ratio=0.1, random_state=42):
    """
    Splits a DataFrame into train and test sets, ensuring that 'part_id' values do not overlap.

    :param df: Input DataFrame containing a "part_id" column.
    :param test_size: Fraction of data to be used for testing.
    :param random_state: Random seed for reproducibility.
    :return: train_df, test_df (DataFrames)
    """
    df = pd.DataFrame().from_records(train_dataset)

    unique_part_ids = df["part_id"].unique()

    # Split part_ids into train and test sets
    train_part_ids, test_part_ids = train_test_split(
        unique_part_ids, test_size=test_ratio, random_state=random_state
    )

    # Select rows where part_id is in the respective sets
    train_df = df[df["part_id"].isin(train_part_ids)]
    test_df = df[df["part_id"].isin(test_part_ids)]

    return train_df.to_dict(orient="records"), test_df.to_dict(orient="records")


def make_prefix(example, template_type):
    """Create prompt prefix based on template type."""
    story_n_question = example["input_text"]

    if template_type == "base":
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {story_n_question}
Please show your reasoning in <think> </think> tags and provide shortest possible final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == "qwen-instruct":
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nStory: {story_n_question}\nPlease show your reasoning in <think> </think> tags and provide shortest possible final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


def process_fantom_data(
    example, idx, split, template_type="base", data_source="fantom"
):
    """Process FANTOM test data into standard format."""
    question = make_prefix(example, template_type)
    solution = {
        "answer": example["correct_answer"],
        "question": example["question"],
        "question_type": example["question_type"],
    }

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "fantom",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": idx,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./data/fantom",
        help="Local directory to save processed data",
    )
    parser.add_argument(
        "--hdfs_dir", type=str, default=None, help="HDFS directory to copy data to"
    )
    parser.add_argument(
        "--fantom_file",
        type=str,
        default="fantom_v1",
        help="Path to rephrased FANTOM JSON file",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Ratio of test examples"
    )
    parser.add_argument(
        "--template_type", type=str, default="base", help="Template type for prompts"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create local directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)

    # Check if FANTOM files exist, if not download from Huggingface
    train_file = os.path.join(args.local_dir, f"{args.fantom_file}.json")

    if not (os.path.exists(train_file)):
        assert FileNotFoundError

    # Load and process datasets
    train_dataset = load_dataset(train_file)
    train_dataset, test_dataset = split_dataset(
        train_dataset, args.test_ratio, random_state=args.seed
    )

    train_dataset, test_dataset = process_dataset(train_dataset), process_dataset(
        test_dataset
    )

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    # Map the processing function over the datasets
    train_dataset = train_dataset.map(
        lambda x, i: process_fantom_data(x, i, "train", args.template_type),
        with_indices=True,
    )
    test_dataset = test_dataset.map(
        lambda x, i: process_fantom_data(x, i, "test", args.template_type),
        with_indices=True,
    )

    # Save processed datasets
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
