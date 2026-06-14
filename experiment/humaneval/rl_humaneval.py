import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION
from experiment.rl_edge_runner import RLDatasetSpec, run_spec, subset_by_task_split
from mas_framework.tools.coding.python_executor import PyExecutor
from mas_framework.tools.reader.readers import JSONLReader


def load_all_records(args):
    return JSONLReader.parse_file(args.dataset_path)


def load_train_records(args):
    return subset_by_task_split(
        load_all_records(args),
        args.task_split_path,
        ["finetune_tasks_indices", "finetune_tasks"],
    )


def load_eval_records(args):
    return subset_by_task_split(load_all_records(args), args.task_split_path, ["test_indices"])


def task_text(record):
    return record["prompt"]


def is_correct(raw_answer, record):
    raw = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer
    code = str(raw).lstrip("```python\n").rstrip("\n```")
    solved, _, _ = PyExecutor().execute(code, [record["test"]], timeout=100)
    return bool(solved)


SPEC = RLDatasetSpec(
    name="humaneval",
    role_constraints=ROLE_DESCRIPTION,
    load_train_records=load_train_records,
    load_eval_records=load_eval_records,
    task_text=task_text,
    is_correct=is_correct,
    default_output_dir="./output/humaneval_rl_edge_model",
    default_dataset_path="../../datasets/humaneval/humaneval-py.jsonl",
    default_task_split_path="./task_split_humaneval.json",
    execution_domain="humaneval",
    default_decision_method="FinalWriteCode",
)


if __name__ == "__main__":
    run_spec(SPEC)
