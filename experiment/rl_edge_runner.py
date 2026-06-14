import argparse
import asyncio
import datetime
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from experiment.mmlu.rl_mmlu import (
    SemanticEntailmentJudge,
    attach_edge_trace_to_test_graph,
    configure_trainable_parameters,
    edge_entropy_rewards,
    edge_semantic_losses,
    execute_graph_with_history,
    sample_graph_with_edge_trace,
    save_rl_checkpoint,
    scale_edge_rewards,
)
from experiment.utils import convert_to_pyg_graph, load_model
from mas_framework.graph.graph import TestGraph
from mas_framework.utils.globals import CompletionTokens, Cost, PromptTokens


@dataclass
class RLDatasetSpec:
    name: str
    role_constraints: Dict[str, str]
    load_train_records: Callable[[argparse.Namespace], List[Any]]
    load_eval_records: Callable[[argparse.Namespace], List[Any]]
    task_text: Callable[[Any], str]
    is_correct: Callable[[Any, Any], bool]
    default_output_dir: str
    default_dataset_path: str
    default_task_split_path: Optional[str]
    execution_domain: str
    default_decision_method: str


def build_parser(spec: RLDatasetSpec):
    parser = argparse.ArgumentParser(
        description=f"Edge-level semantic entropy RL stage for ARGDesigner on {spec.name}."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing best_model.pth or ef_best_model.pth.")
    parser.add_argument("--load_ef", action="store_true",
                        help="Load ef_best_model.pth instead of best_model.pth.")
    parser.add_argument("--output_dir", type=str, default=spec.default_output_dir)
    parser.add_argument("--dataset_path", type=str, default=spec.default_dataset_path)
    if spec.default_task_split_path is not None:
        parser.add_argument("--task_split_path", type=str, default=spec.default_task_split_path)
    else:
        parser.add_argument("--task_split_path", type=str, default="")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--domain", type=str, default=spec.execution_domain)
    parser.add_argument("--decision_method", type=str, default=spec.default_decision_method)
    parser.add_argument("--embedding_model", type=str, default="/Models/all-MiniLM-L6-v2")
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--samples_per_prompt", type=int, default=1,
                        help="Number of graph trajectories sampled for each task prompt per RL iteration.")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--limit_questions", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--edge_epsilon", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--semantic_lambda", type=float, default=1.0)
    parser.add_argument("--sparsity_penalty", type=float, default=0.0,
                        help="Deprecated compatibility option; sparsity penalty is not used.")
    parser.add_argument("--edge_reward_clip", type=float, default=5.0,
                        help="Clip scaled edge semantic advantages to [-value, value]. 0 disables clipping.")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="KL regularization coefficient against the frozen initial edge policy. 0 disables KL.")
    parser.add_argument("--num_entropy_samples", type=int, default=2)
    parser.add_argument("--negative_edge_reward_scale", type=float, default=1.0,
                        help="Deprecated compatibility option; raw signed entropy gain is used.")
    parser.add_argument("--nonpositive_edge_penalty", type=float, default=0.01,
                        help="Deprecated compatibility option; raw signed entropy gain is used.")
    parser.add_argument("--train_node_context", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_edge_epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_judge_llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--semantic_judge_api_key", type=str, default="")
    parser.add_argument("--semantic_judge_base_url", type=str, default="")
    parser.add_argument("--semantic_judge_model_path", type=str, default="")
    parser.add_argument("--semantic_judge_max_concurrency", type=int, default=None)
    return parser


def subset_by_task_split(
    records: List[Any],
    task_split_path: str,
    keys: List[str],
) -> List[Any]:
    if not task_split_path or not os.path.exists(task_split_path):
        return records
    with open(task_split_path, "r", encoding="utf-8") as file:
        split = json.load(file)
    for key in keys:
        indices = split.get(key)
        if indices:
            return [records[i] for i in indices if i < len(records)]
    return records


def infinite_loader(records: List[Any], limit_questions: Optional[int]):
    items = records[:limit_questions] if limit_questions is not None else list(records)
    if not items:
        raise ValueError("No records available for RL training.")
    while True:
        random.shuffle(items)
        for item in items:
            yield item


async def run_one_sample(
    model,
    sentence_model,
    spec: RLDatasetSpec,
    args,
    record,
    judge: Optional[SemanticEntailmentJudge],
    train_mode: bool,
    ref_model=None,
):
    task_text = spec.task_text(record)
    task_embedding = torch.tensor(
        sentence_model.encode(task_text),
        device=model.args.device,
        dtype=torch.float32,
    )
    generated_graph, trace = sample_graph_with_edge_trace(
        model,
        task_embedding,
        spec.role_constraints,
        edge_epsilon=args.edge_epsilon if train_mode else args.eval_edge_epsilon,
        train_node_context=args.train_node_context if train_mode else False,
        ref_model=ref_model if train_mode else None,
    )
    pyg_graph = convert_to_pyg_graph(generated_graph, task_text)
    test_graph = TestGraph(
        domain=args.domain,
        llm_name=args.llm_name,
        decision_method=args.decision_method,
        pyg_data=pyg_graph,
    )
    attach_edge_trace_to_test_graph(test_graph, trace["edge_log_probs"], args.num_rounds)
    input_dict = {"task": task_text}

    if train_mode:
        raw_answer = await execute_graph_with_history(
            test_graph,
            input_dict,
            args.num_rounds,
            max(2, int(args.num_entropy_samples)),
        )
    else:
        raw_answer = await test_graph.arun(input_dict, args.num_rounds)

    is_correct = spec.is_correct(raw_answer, record)
    edge_rewards: Dict[str, float] = {}
    edge_details: Dict[str, Dict[str, Any]] = {}
    if train_mode and is_correct and judge is not None:
        edge_rewards, edge_details = await edge_entropy_rewards(
            test_graph,
            task_text,
            input_dict,
            judge,
            max(2, int(args.num_entropy_samples)),
            negative_reward_scale=args.negative_edge_reward_scale,
            nonpositive_penalty=args.nonpositive_edge_penalty,
        )

    loss = torch.tensor(0.0, device=model.args.device)
    kl_value = 0.0
    if train_mode and is_correct:
        edge_losses = edge_semantic_losses(
            test_graph.edge_log_probs,
            edge_rewards,
            semantic_lambda=args.semantic_lambda,
            sparsity_penalty=args.sparsity_penalty,
            correctness_reward=1.0 if is_correct else 0.0,
            edge_reward_clip=args.edge_reward_clip,
        )
        if edge_losses:
            loss = torch.stack(edge_losses).sum()
        if trace["edge_entropies"]:
            entropy_bonus = torch.stack(trace["edge_entropies"]).mean()
            loss = loss - args.entropy_coef * entropy_bonus
        if args.kl_coef > 0 and trace.get("edge_kls"):
            kl_loss = torch.stack(trace["edge_kls"]).mean()
            loss = loss + args.kl_coef * kl_loss
            kl_value = float(kl_loss.detach().cpu())

    return {
        "loss": loss,
        "is_correct": is_correct,
        "raw_answer": raw_answer,
        "num_edges": generated_graph.number_of_edges(),
        "edge_rewards": edge_rewards,
        "scaled_edge_rewards": scale_edge_rewards(edge_rewards, args.edge_reward_clip),
        "edge_details": edge_details,
        "kl_value": kl_value,
    }


async def evaluate_current_generator(model, sentence_model, spec: RLDatasetSpec, args, records, iteration):
    model.eval()
    total = min(args.eval_batch_size, len(records))
    if total <= 0:
        print(f"[{spec.name} eval] skipped: empty eval records")
        model.train()
        return

    start_cost = Cost.instance().value
    start_prompt = PromptTokens.instance().value
    start_completion = CompletionTokens.instance().value
    start_ts = time.time()

    correct = 0
    edge_counts = []
    with torch.no_grad():
        for record in records[:total]:
            try:
                result = await run_one_sample(
                    model, sentence_model, spec, args, record, judge=None, train_mode=False
                )
                correct += int(result["is_correct"])
                edge_counts.append(result["num_edges"])
            except Exception as exc:
                print(f"[{spec.name} eval iter {iteration}] sample failed: {exc}")

    cost = Cost.instance().value - start_cost
    prompt_tokens = PromptTokens.instance().value - start_prompt
    completion_tokens = CompletionTokens.instance().value - start_completion
    avg_edges = sum(edge_counts) / max(1, len(edge_counts))
    accuracy = correct / max(1, total)
    print(
        f"[{spec.name} eval iter {iteration}] "
        f"accuracy={accuracy:.3f} ({correct}/{total}) "
        f"avg_edges={avg_edges:.2f} "
        f"cost=${cost:.6f} "
        f"prompt_tokens={int(prompt_tokens)} "
        f"completion_tokens={int(completion_tokens)} "
        f"time={time.time() - start_ts:.1f}s"
    )
    model.train()


async def train_edge_rl(spec: RLDatasetSpec, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = load_model(args.model_path, ef=args.load_ef)
    model.train()
    ref_model = None
    if args.kl_coef > 0:
        ref_model = load_model(args.model_path, ef=args.load_ef)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    trainable_params = configure_trainable_parameters(model, args.train_node_context)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    sentence_model = SentenceTransformer(args.embedding_model)

    train_records = spec.load_train_records(args)
    eval_records = spec.load_eval_records(args) if args.eval_every > 0 else []
    loader = infinite_loader(train_records, args.limit_questions)

    judge = SemanticEntailmentJudge(
        llm_name=args.semantic_judge_llm_name,
        api_key=args.semantic_judge_api_key,
        base_url=args.semantic_judge_base_url,
        model_path=args.semantic_judge_model_path,
        max_concurrency=args.semantic_judge_max_concurrency,
    )
    if not judge.is_configured:
        raise RuntimeError(
            "Semantic judge is not configured. Provide --semantic_judge_api_key, "
            "OPENAI_API_KEY, or --semantic_judge_base_url for local vLLM."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "rl_metrics.jsonl")
    best_correct_rate = -1.0

    for iteration in range(args.num_iterations):
        start_ts = time.time()
        losses = []
        correctness = []
        edge_counts = []
        reward_summaries = []
        kl_values = []

        records = [next(loader) for _ in range(args.batch_size)]
        samples_per_prompt = max(1, int(args.samples_per_prompt))
        for batch_idx, record in enumerate(records):
            task_text = spec.task_text(record)
            for sample_idx in range(samples_per_prompt):
                try:
                    result = await run_one_sample(
                        model,
                        sentence_model,
                        spec,
                        args,
                        record,
                        judge=judge,
                        train_mode=True,
                        ref_model=ref_model,
                    )
                except Exception as exc:
                    print(
                        f"[{spec.name} iter {iteration + 1} item {batch_idx + 1} "
                        f"sample {sample_idx + 1}/{samples_per_prompt}] failed: {exc}"
                    )
                    continue

                if result["loss"].requires_grad:
                    losses.append(result["loss"])
                correctness.append(1.0 if result["is_correct"] else 0.0)
                edge_counts.append(result["num_edges"])
                if result["kl_value"] > 0:
                    kl_values.append(result["kl_value"])
                reward_summaries.append({
                    "task": task_text,
                    "sample_idx": sample_idx,
                    "correct": result["is_correct"],
                    "num_edges": result["num_edges"],
                    "edge_rewards": result["edge_rewards"],
                    "scaled_edge_rewards": result["scaled_edge_rewards"],
                    "edge_details": result["edge_details"],
                })
                print(
                    f"[{spec.name} iter {iteration + 1} item {batch_idx + 1} "
                    f"sample {sample_idx + 1}/{samples_per_prompt}] "
                    f"correct={result['is_correct']} edges={result['num_edges']}"
                )
                if result["edge_rewards"]:
                    print(f"  edge rewards: {result['edge_rewards']}")
                    print(f"  scaled edge rewards: {result['scaled_edge_rewards']}")

        if losses:
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            loss_value = float(total_loss.detach().cpu())
        else:
            loss_value = 0.0

        correct_rate = sum(correctness) / max(1, len(correctness))
        avg_edges = sum(edge_counts) / max(1, len(edge_counts))
        metric = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": spec.name,
            "iteration": iteration + 1,
            "loss": loss_value,
            "correct_rate": correct_rate,
            "avg_edges": avg_edges,
            "avg_kl": sum(kl_values) / max(1, len(kl_values)),
            "samples_per_prompt": samples_per_prompt,
            "reward_summaries": reward_summaries,
            "cost": Cost.instance().value,
            "prompt_tokens": PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value,
            "elapsed_sec": time.time() - start_ts,
        }
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(metric, ensure_ascii=False, default=str) + "\n")

        print(
            f"{spec.name} Iter {iteration + 1}/{args.num_iterations}: "
            f"loss={loss_value:.4f} correct_rate={correct_rate:.3f} "
            f"avg_edges={avg_edges:.2f} "
            f"avg_kl={sum(kl_values) / max(1, len(kl_values)):.6f} "
            f"time={time.time() - start_ts:.1f}s"
        )

        if correct_rate > best_correct_rate:
            best_correct_rate = correct_rate
            path = save_rl_checkpoint(model, args.output_dir, args, "ef_best_model.pth")
            save_rl_checkpoint(model, args.output_dir, args, "rl_best_model.pth")
            print(f"Saved best RL checkpoint to {path}")

        if args.save_every > 0 and (iteration + 1) % args.save_every == 0:
            path = save_rl_checkpoint(model, args.output_dir, args, f"rl_iter_{iteration + 1}.pth")
            print(f"Saved periodic RL checkpoint to {path}")

        if args.eval_every > 0 and (iteration + 1) % args.eval_every == 0:
            await evaluate_current_generator(
                model, sentence_model, spec, args, eval_records, iteration + 1
            )

    final_path = save_rl_checkpoint(model, args.output_dir, args, "rl_final_model.pth")
    print(f"{spec.name} edge RL stage complete. Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")


def run_spec(spec: RLDatasetSpec):
    args = build_parser(spec).parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(train_edge_rl(spec, args))
