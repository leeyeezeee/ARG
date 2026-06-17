import os
import json
import math
import asyncio
import time
import torch
import numpy as np
import argparse
import random
from tqdm import tqdm
from typing import List, Any, Dict, Iterator
import sys
import csv
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mas_framework.utils.globals import Cost, PromptTokens, CompletionTokens
from sentence_transformers import SentenceTransformer
from mas_framework.graph.graph import TestGraph
from experiment.utils import Accuracy, load_model, generate_graph, convert_to_pyg_graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU")
    parser.add_argument('--agent_nums', type=int, default=5,
                        help="number of agents")
    parser.add_argument('--data_dir', type=str, default='../../ColdStartData_mmlu',
                        help="data directory")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="number of inference rounds per query")
    parser.add_argument('--llm_name', type=str, default="qwen3-8b",
                        help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="dataset name")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="limit number of questions to evaluate")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="decision method for final node")
    parser.add_argument('--model_path', type=str, default='output/xxx',
                        help="path to pretrained model")
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help="evaluation batch size")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--embedding_model', type=str, default="/data/lyz/models/all-MiniLM-L6-v2",
                        help="model for task embeddings")
    parser.add_argument('--summary_log_file', type=str, default='./res_logs/evaluation_summary.jsonl',
                        help="log file to record evaluation summaries")
    parser.add_argument('--wrong_samples_file', type=str, default='./res_logs/mmlu_wrong_samples.jsonl',
                        help="JSONL file to save wrong MMLU samples with generated graph and outputs")

    return parser.parse_args()


def serialize_generated_graph(graph) -> Dict[str, Any]:
    nodes = []
    for node_id, data in graph.nodes(data=True):
        nodes.append({
            "id": int(node_id) if isinstance(node_id, (int, np.integer)) else str(node_id),
            "role": data.get("role", "Unknown"),
            "constraint": data.get("constraint", ""),
            "label": data.get("label"),
        })

    edges = []
    for src, dst, data in graph.edges(data=True):
        src_role = graph.nodes[src].get("role", "Unknown")
        dst_role = graph.nodes[dst].get("role", "Unknown")
        edges.append({
            "source": int(src) if isinstance(src, (int, np.integer)) else str(src),
            "target": int(dst) if isinstance(dst, (int, np.integer)) else str(dst),
            "source_role": src_role,
            "target_role": dst_role,
            "label": data.get("label"),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "mode": graph.graph.get("mode"),
            "agent_nums": graph.graph.get("agent_nums"),
            "roles": graph.graph.get("roles"),
            "sampled_node_types": graph.graph.get("sampled_node_types"),
        }
    }


def serialize_test_graph_outputs(test_graph: TestGraph) -> Dict[str, Any]:
    nodes = []
    for idx, node in enumerate(test_graph.nodes.values()):
        nodes.append({
            "index": idx,
            "id": node.id,
            "role": getattr(node, "role", ""),
            "constraint": getattr(node, "constraint", ""),
            "outputs": getattr(node, "outputs", []),
            "spatial_predecessors": [
                {"id": predecessor.id, "role": getattr(predecessor, "role", "")}
                for predecessor in getattr(node, "spatial_predecessors", [])
            ],
            "spatial_successors": [
                {"id": successor.id, "role": getattr(successor, "role", "")}
                for successor in getattr(node, "spatial_successors", [])
            ],
        })

    decision_node = test_graph.decision_node
    return {
        "nodes": nodes,
        "decision_node": {
            "id": decision_node.id,
            "role": getattr(decision_node, "role", ""),
            "constraint": getattr(decision_node, "constraint", ""),
            "outputs": getattr(decision_node, "outputs", []),
        }
    }


def append_wrong_sample(record: Dict[str, Any], path: str):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


async def evaluate(
        model,
        dataset,
        sentence_model,
        role_constraints_dict,
        args
) -> Dict[str, float]:
    """Evaluate model on MMLU dataset"""
    print(f"Evaluating model on {dataset.__class__.__name__}")

    accuracy = Accuracy()
    limit_questions = args.limit_questions

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / args.eval_batch_size))
    total_nodes = 0
    total_edges = 0
    graph_count = 0

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=args.eval_batch_size)), total=num_batches):
        print(f"{'-' * 80}")

        start_ts = time.time()
        answer_tasks = []
        questions = []
        generated_graphs = []
        test_graphs = []
        question_ids = []

        for i, record in enumerate(record_batch):
            input_dict = dataset.record_to_input(record)
            task_text = input_dict['task']
            questions.append(task_text)

            question_id = i_batch * args.eval_batch_size + i + 1
            question_ids.append(question_id)

            task_embedding = torch.tensor(
                sentence_model.encode(task_text),
                device=model.args.device
            ).float()

            generated_graph = generate_graph(
                model,
                task_embedding,
                role_constraints_dict,
                question_id=question_id
            )
            generated_graphs.append(generated_graph[0])
            total_nodes += generated_graph[0].number_of_nodes()
            total_edges += generated_graph[0].number_of_edges()
            graph_count += 1

            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=convert_to_pyg_graph(generated_graph[0], task_text)
            )
            test_graphs.append(tg)
            answer_tasks.append(asyncio.create_task(tg.arun(input_dict, args.num_rounds)))

        raw_results = await asyncio.gather(*answer_tasks, return_exceptions=True)
        is_corrects = []

        for i, (raw_answer, record, graph, test_graph) in enumerate(
                zip(raw_results, record_batch, generated_graphs, test_graphs)):
            correct_answer = dataset.record_to_target_answer(record)
            error = None
            if isinstance(raw_answer, Exception):
                print(f"LLM error for question: {raw_answer}")
                answer = ""
                error = str(raw_answer)
                is_correct = accuracy.update(answer, correct_answer)
            else:
                try:
                    answer = dataset.postprocess_answer(raw_answer)
                except Exception as exc:
                    answer = ""
                    error = f"postprocess error: {exc}"
                is_correct = accuracy.update(answer, correct_answer)

            current_accuracy = accuracy.get() * 100
            print(f"Accuracy: {current_accuracy:.1f}% | "
                  f"Cost: ${Cost.instance().value:.4f} | "
                  f"Tokens: P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})")

            is_corrects.append(is_correct)
            if not is_correct:
                wrong_sample = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "dataset": "mmlu",
                    "id": i_batch * args.eval_batch_size + i,
                    "question_id": question_ids[i],
                    "question": record.get("question", ""),
                    "options": {
                        "A": record.get("A", ""),
                        "B": record.get("B", ""),
                        "C": record.get("C", ""),
                        "D": record.get("D", ""),
                    },
                    "task": questions[i],
                    "correct_answer": correct_answer,
                    "predicted_answer": answer,
                    "raw_output": None if isinstance(raw_answer, Exception) else raw_answer,
                    "error": error,
                    "generated_graph": serialize_generated_graph(graph),
                    "execution_outputs": serialize_test_graph_outputs(test_graph),
                }
                append_wrong_sample(wrong_sample, args.wrong_samples_file)

        for i in range(len(record_batch)):
            batch_data = [{
                "dataset": "mmlu",
                "id": i_batch * args.eval_batch_size + i,
                "question": questions[i],
                "mode": "ARGDesigner",
                "size": args.agent_nums,
                "is_correct": is_corrects[i]
            }]
            write_to_csv(batch_data)

        print(f"Batch time: {time.time() - start_ts:.3f}s")

    accuracy.print()
    print("Evaluation complete!")
    print(f"Wrong samples saved to: {args.wrong_samples_file}")
    return {
        "accuracy": accuracy.get(),
        "avg_num_nodes": total_nodes / graph_count if graph_count else 0.0,
        "avg_num_edges": total_edges / graph_count if graph_count else 0.0,
    }


def write_to_csv(
        data: List[Dict],
        filename: str = "ARGDesigner_results.csv",
        fieldnames: List[str] = ["dataset", "id", "question", "mode", "size", "is_correct"]
):
    """Write result data to CSV file"""
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(data)


async def main(ef=True):
    args = parse_args()

    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("Loading MMLU dataset...")
    # download()
    dataset_test = MMLUDataset('val')

    print(f"Loading Sentence Transformer model: {args.embedding_model}")
    sentence_model = SentenceTransformer(args.embedding_model)

    from mmlu_prompt_set import ROLE_DESCRIPTION
    role_constraints_dict = {role: desc for role, desc in ROLE_DESCRIPTION.items()}

    print(f"Loading pretrained model: {args.model_path}")
    args.model_name = 'ef_best' if ef else 'best'
    print('model_name', args.model_name)
    model = load_model(args.model_path, ef=ef)

    eval_metrics = await evaluate(
        model=model,
        dataset=dataset_test,
        sentence_model=sentence_model,
        role_constraints_dict=role_constraints_dict,
        args=args
    )
    score = eval_metrics["accuracy"]
    avg_num_nodes = eval_metrics["avg_num_nodes"]
    avg_num_edges = eval_metrics["avg_num_edges"]

    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value
    total_tasks = min(len(dataset_test), args.limit_questions) if args.limit_questions is not None else len(dataset_test)

    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Model path: {args.model_path}")
    print(f"Total tasks: {total_tasks}\nFinal accuracy (Pass@1): {score:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print(f"Average nodes: {avg_num_nodes:.2f}")
    print(f"Average edges: {avg_num_edges:.2f}")
    print("-" * 50)
    print("Detailed CSV results saved to: ARGDesigner_results.csv")

    log_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "mmlu",
        "model_path": os.path.join(args.model_path, args.model_name),
        "llm_name": args.llm_name,
        "total_tasks": total_tasks,
        "pass_at_1": score,
        "cost": final_cost,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "avg_num_nodes": avg_num_nodes,
        "avg_num_edges": avg_num_edges,
        "detail_file": "ARGDesigner_results.csv",
        "wrong_samples_file": args.wrong_samples_file,
    }

    try:
        os.makedirs(os.path.dirname(args.summary_log_file), exist_ok=True)
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + '\n')
        print(f"Summary log appended to: {args.summary_log_file}")
    except Exception as e:
        print(f"Failed to write summary log file: {e}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main(True))
