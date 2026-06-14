import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
from experiment.rl_edge_runner import RLDatasetSpec, run_spec, subset_by_task_split
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from mas_framework.tools.reader.readers import JSONLReader


def load_all_records(args):
    raw = JSONLReader.parse_file(args.dataset_path)
    return aqua_data_process(raw)


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
    return aqua_get_predict(raw) == record["answer"]


SPEC = RLDatasetSpec(
    name="aqua",
    role_constraints=ROLE_DESCRIPTION,
    load_train_records=load_train_records,
    load_eval_records=load_eval_records,
    task_text=task_text,
    is_correct=is_correct,
    default_output_dir="./output/aqua_rl_edge_model",
    default_dataset_path="../../datasets/AQuA/AQuA.jsonl",
    default_task_split_path="./task_split_aqua.json",
    execution_domain="aqua",
    default_decision_method="FinalRefer",
)


if __name__ == "__main__":
    run_spec(SPEC)
