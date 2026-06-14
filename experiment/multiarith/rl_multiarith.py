import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiment.multiarith.gsm8k_prompt_set import ROLE_DESCRIPTION
from experiment.rl_edge_runner import RLDatasetSpec, run_spec, subset_by_task_split
from datasets.gsm8k_dataset import gsm_get_predict, multiarith_data_process
from mas_framework.tools.reader.readers import JSONReader


def load_all_records(args):
    raw = JSONReader.parse_file(args.dataset_path)
    return multiarith_data_process(raw)


def load_train_records(args):
    return subset_by_task_split(
        load_all_records(args),
        args.task_split_path,
        ["finetune_tasks_indices", "finetune_tasks"],
    )


def load_eval_records(args):
    return subset_by_task_split(load_all_records(args), args.task_split_path, ["test_indices"])


def task_text(record):
    return record["task"]


def is_correct(raw_answer, record):
    raw = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer
    pred = gsm_get_predict(raw)
    try:
        return float(pred) == float(record["answer"])
    except (TypeError, ValueError):
        return False


SPEC = RLDatasetSpec(
    name="multiarith",
    role_constraints=ROLE_DESCRIPTION,
    load_train_records=load_train_records,
    load_eval_records=load_eval_records,
    task_text=task_text,
    is_correct=is_correct,
    default_output_dir="./output/multiarith_rl_edge_model",
    default_dataset_path="../../datasets/MultiArith/MultiArith.json",
    default_task_split_path="./task_split_multiarith.json",
    execution_domain="gsm8k",
    default_decision_method="FinalRefer",
)


if __name__ == "__main__":
    run_spec(SPEC)
