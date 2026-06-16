import argparse
import asyncio
import datetime
import json
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
from experiment.utils import convert_to_pyg_graph, load_model
from datasets.mmlu_dataset import MMLUDataset
from mas_framework.graph.graph import TestGraph
from mas_framework.utils.globals import CompletionTokens, Cost, PromptTokens


T = TypeVar("T")
EPS = 1e-8
PROB_EPS = 1e-6

RL_FINAL_CHECKPOINT = "ef_best_model.pth"
WRONG_EDGE_REWARD = -1.0

DEFAULT_JUDGE_TIMEOUT = 120.0
DEFAULT_JUDGE_CONNECT_TIMEOUT = 10.0
DEFAULT_JUDGE_MAX_RETRIES = 3
DEFAULT_JUDGE_MAX_CONCURRENCY = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge-level RL stage for ARGDesigner on MMLU."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing best_model.pth or ef_best_model.pth.")
    parser.add_argument("--load_ef", action="store_true",
                        help="Load ef_best_model.pth instead of best_model.pth.")
    parser.add_argument("--output_dir", type=str, default="./output/mmlu_rl_edge_model")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--embedding_model", type=str, default="/Models/all-MiniLM-L6-v2")
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--samples_per_prompt", type=int, default=1,
                        help="Number of graph trajectories sampled for each task prompt per RL iteration.")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--limit_questions", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--edge_epsilon", type=float, default=0.15,
                        help="Mix edge probabilities with 0.5 for exploration.")
    parser.add_argument("--entropy_coef", type=float, default=1e-3,
                        help="Entropy bonus for candidate edge decisions.")
    parser.add_argument("--semantic_lambda", type=float, default=1.0,
                        help="Scale for each edge's semantic entropy reward.")
    parser.add_argument("--sparsity_penalty", type=float, default=0.0,
                        help="Deprecated compatibility option; sparsity penalty is not used.")
    parser.add_argument("--edge_reward_clip", type=float, default=5.0,
                        help="Clip scaled edge semantic advantages to [-value, value]. 0 disables clipping.")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="KL regularization coefficient against the frozen initial edge policy. 0 disables KL.")
    parser.add_argument("--num_entropy_samples", type=int, default=2,
                        help="Samples per target node for semantic entropy. Must be >=2.")
    parser.add_argument("--negative_edge_reward_scale", type=float, default=1.0,
                        help="Deprecated compatibility option; raw signed entropy gain is used.")
    parser.add_argument("--nonpositive_edge_penalty", type=float, default=0.01,
                        help="Deprecated compatibility option; raw signed entropy gain is used.")
    parser.add_argument("--train_node_context", action="store_true",
                        help="Allow edge loss gradients into node context layers at the same LR.")
    parser.add_argument("--feed_previous_edge_features_to_node", action="store_true",
                        help="Feed sampled previous edge features into subsequent node generation during RL rollout.")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Deprecated compatibility option; per-iteration RL checkpoints are not saved.")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate the current generator every N RL iterations. 0 disables periodic eval.")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Number of validation samples to evaluate at each periodic eval.")
    parser.add_argument("--eval_split", type=str, default="val",
                        help="MMLU split used for periodic evaluation.")
    parser.add_argument("--eval_edge_epsilon", type=float, default=0.0,
                        help="Exploration epsilon used only during periodic evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_judge_llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--semantic_judge_api_key", type=str, default="")
    parser.add_argument("--semantic_judge_base_url", type=str, default="")
    parser.add_argument("--semantic_judge_model_path", type=str, default="")
    parser.add_argument("--semantic_judge_max_concurrency", type=int, default=None)
    return parser.parse_args()


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def semantic_entropy(labels: Iterable[str]) -> float:
    valid_labels = [label for label in labels if label]
    if len(valid_labels) <= 1:
        return 0.0

    counts = Counter(valid_labels)
    total = len(valid_labels)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def _semantic_judge_extra_body(model: str) -> Dict[str, Any]:
    if "qwen" not in model.lower():
        return {}
    return {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class SemanticEntailmentJudge:
    def __init__(
        self,
        llm_name: Optional[str] = None,
        api_key: str = "",
        base_url: str = "",
        model_path: str = "",
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_concurrency: Optional[int] = None,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        self.timeout = timeout if timeout is not None else _float_env(
            "SEMANTIC_JUDGE_TIMEOUT", DEFAULT_JUDGE_TIMEOUT
        )
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else _float_env("SEMANTIC_JUDGE_CONNECT_TIMEOUT", DEFAULT_JUDGE_CONNECT_TIMEOUT)
        )
        self.max_retries = max_retries if max_retries is not None else _int_env(
            "SEMANTIC_JUDGE_MAX_RETRIES", DEFAULT_JUDGE_MAX_RETRIES
        )
        self.max_concurrency = max(
            1,
            max_concurrency
            if max_concurrency is not None
            else _int_env("SEMANTIC_JUDGE_MAX_CONCURRENCY", DEFAULT_JUDGE_MAX_CONCURRENCY),
        )
        self._request_semaphore = asyncio.Semaphore(self.max_concurrency)

        self.llm_name = (
            model_path
            or llm_name
            or os.getenv("SEMANTIC_JUDGE_MODEL")
            or "gpt-4o-mini"
        )
        self.api_key = (
            api_key
            or os.getenv("SEMANTIC_JUDGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.getenv("SEMANTIC_JUDGE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        )
        if self.base_url and not self.api_key:
            self.api_key = "EMPTY"

        self._client = None
        if self.llm_name and self.api_key:
            from httpx import Timeout
            from openai import AsyncOpenAI

            kwargs = {
                "api_key": self.api_key,
                "timeout": Timeout(timeout=self.timeout, connect=self.connect_timeout),
                "max_retries": 0,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)

    @property
    def is_configured(self) -> bool:
        return bool(self._client and self.llm_name)

    async def _create_completion(self, request_kwargs: Dict[str, Any]):
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
        from tenacity import wait_random_exponential

        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(max(1, self.max_retries)),
            retry=retry_if_exception_type(
                (APITimeoutError, APIConnectionError, RateLimitError)
            ),
            reraise=True,
        ):
            with attempt:
                async with self._request_semaphore:
                    return await self._client.chat.completions.create(**request_kwargs)

    async def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        if self._client is None:
            raise RuntimeError(
                "SemanticEntailmentJudge is not configured. Set OPENAI_API_KEY, "
                "SEMANTIC_JUDGE_API_KEY, or --semantic_judge_base_url for local vLLM."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict natural language inference judge. Decide whether "
                    "the premise entails the hypothesis for the given task. Return only "
                    "one token: entailment, contradiction, or neutral."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{question}\n\n"
                    f"Premise:\n{premise}\n\n"
                    f"Hypothesis:\n{hypothesis}\n\n"
                    "Does the premise entail the hypothesis?"
                ),
            },
        ]
        request_kwargs: Dict[str, Any] = {
            "model": self.llm_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 32,
        }
        extra_body = _semantic_judge_extra_body(self.llm_name)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = await self._create_completion(request_kwargs)
        verdict = response.choices[0].message.content or ""
        return verdict.strip().lower().startswith("entail")

    async def equivalent(self, question: str, output_a: str, output_b: str) -> bool:
        if output_a.strip() == output_b.strip():
            return True
        forward, backward = await asyncio.gather(
            self.entails(question, output_a, output_b),
            self.entails(question, output_b, output_a),
        )
        return bool(forward and backward)

    async def cluster_outputs(self, question: str, outputs: Iterable[Any]) -> List[str]:
        valid_outputs = [str(output) for output in outputs if str(output).strip()]
        clusters: List[List[str]] = []
        labels: List[str] = []
        for output in valid_outputs:
            label = ""
            comparisons = []
            if clusters:
                comparisons = await asyncio.gather(
                    *[self.equivalent(question, output, cluster[0]) for cluster in clusters]
                )
            for cluster_idx, equivalent in enumerate(comparisons):
                if equivalent:
                    clusters[cluster_idx].append(output)
                    label = f"cluster_{cluster_idx}"
                    break
            if not label:
                clusters.append([output])
                label = f"cluster_{len(clusters) - 1}"
            labels.append(label)
        return labels


async def semantic_uncertainty(
    question: str,
    outputs: Iterable[T],
    judge: SemanticEntailmentJudge,
) -> Tuple[float, List[str]]:
    labels = await judge.cluster_outputs(question, outputs)
    return semantic_entropy(labels), labels


def edge_key(edge_info: Dict[str, Any]) -> str:
    return edge_info.get(
        "edge_key",
        f"{edge_info['type']}:{edge_info['round']}:{edge_info['source']}->{edge_info['target']}",
    )


def _flatten_outputs(results: Iterable[Any]) -> List[Any]:
    outputs = []
    for result in results:
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
    return outputs


def _as_output_list(result: Any) -> List[Any]:
    return result if isinstance(result, list) else [result]


async def _sample_node_outputs(
    node,
    input_data: Any,
    spatial_info: Dict[str, Any],
    temporal_info: Dict[str, Any],
    num_samples: int,
) -> List[Any]:
    tasks = [
        asyncio.create_task(node._async_execute(input_data, spatial_info, temporal_info))
        for _ in range(max(1, int(num_samples)))
    ]
    return _flatten_outputs(await asyncio.gather(*tasks, return_exceptions=False))


def _edge_reward_from_delta(
    entropy_delta: float,
    negative_reward_scale: float,
    nonpositive_penalty: float,
) -> float:
    # Keep the semantic signal as the raw signed entropy gain. Scaling and
    # sparsity penalties are applied later when building the policy-gradient
    # advantage, so the sign still means exactly: positive helps, negative hurts.
    return entropy_delta


def scale_edge_rewards(
    edge_rewards: Dict[str, float],
    clip_value: float = 5.0,
) -> Dict[str, float]:
    if not edge_rewards:
        return {}

    rewards = torch.tensor(list(edge_rewards.values()), dtype=torch.float32)
    if not torch.isfinite(rewards).all():
        raise ValueError(f"Non-finite edge rewards: {edge_rewards}")
    abs_rewards = rewards.abs()
    scale = abs_rewards.std(unbiased=False)
    if float(scale) < EPS:
        scale = abs_rewards.mean()
    if float(scale) < EPS:
        scale = torch.tensor(1.0, dtype=torch.float32)

    scaled = rewards / (scale + EPS)
    if clip_value and clip_value > 0:
        scaled = scaled.clamp(-float(clip_value), float(clip_value))

    return {
        key: float(value)
        for key, value in zip(edge_rewards.keys(), scaled.tolist())
    }


def bernoulli_kl(current_prob: torch.Tensor, ref_prob: torch.Tensor) -> torch.Tensor:
    p = current_prob.clamp(PROB_EPS, 1 - PROB_EPS)
    q = ref_prob.detach().clamp(PROB_EPS, 1 - PROB_EPS)
    kl = p * (p.log() - q.log()) + (1 - p) * (
        (1 - p).log() - (1 - q).log()
    )
    if not torch.isfinite(kl).all():
        raise ValueError(f"Non-finite Bernoulli KL: p={p}, q={q}, kl={kl}")
    return kl


async def edge_entropy_rewards(
    graph,
    question: str,
    input_data: Any,
    judge: SemanticEntailmentJudge,
    num_entropy_samples: int,
    negative_reward_scale: float = 1.0,
    nonpositive_penalty: float = 0.01,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    if not getattr(graph, "edge_log_probs", None) or num_entropy_samples <= 1:
        return {}, {}

    histories: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        for history_item in getattr(node, "execution_history", []):
            histories[(node_id, history_item["round"])] = history_item

    rewards: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    after_cache: Dict[Tuple[str, int], Tuple[float, List[str]]] = {}

    for edge_info in graph.edge_log_probs:
        target_id = edge_info["target"]
        source_id = edge_info["source"]
        round_idx = edge_info["round"]
        edge_type = edge_info["type"]
        key = edge_key(edge_info)
        history_item = histories.get((target_id, round_idx))
        target_node = graph.nodes.get(target_id)
        if history_item is None or target_node is None:
            continue

        spatial_info = {
            node_id: dict(info)
            for node_id, info in history_item.get("spatial_info", {}).items()
        }
        temporal_info = {
            node_id: dict(info)
            for node_id, info in history_item.get("temporal_info", {}).items()
        }
        if edge_type == "spatial":
            if source_id not in spatial_info:
                continue
            before_spatial_info = dict(spatial_info)
            before_temporal_info = temporal_info
            before_spatial_info.pop(source_id, None)
        elif edge_type == "temporal":
            if source_id not in temporal_info:
                continue
            before_spatial_info = spatial_info
            before_temporal_info = dict(temporal_info)
            before_temporal_info.pop(source_id, None)
        else:
            continue

        before_outputs = await _sample_node_outputs(
            target_node,
            input_data,
            before_spatial_info,
            before_temporal_info,
            num_entropy_samples,
        )
        after_outputs = history_item.get("entropy_samples", [])
        if not before_outputs or not after_outputs:
            continue

        after_cache_key = (target_id, round_idx)
        if after_cache_key in after_cache:
            before_entropy, before_labels = await semantic_uncertainty(
                question, before_outputs, judge
            )
            after_entropy, after_labels = after_cache[after_cache_key]
        else:
            before_result, after_result = await asyncio.gather(
                semantic_uncertainty(question, before_outputs, judge),
                semantic_uncertainty(question, after_outputs, judge),
            )
            before_entropy, before_labels = before_result
            after_entropy, after_labels = after_result
            after_cache[after_cache_key] = (after_entropy, after_labels)

        entropy_delta = before_entropy - after_entropy
        reward = _edge_reward_from_delta(
            entropy_delta,
            negative_reward_scale=negative_reward_scale,
            nonpositive_penalty=nonpositive_penalty,
        )
        rewards[key] = reward
        details[key] = {
            "type": edge_type,
            "round": round_idx,
            "source": source_id,
            "target": target_id,
            "before_entropy": before_entropy,
            "after_entropy": after_entropy,
            "entropy_delta": entropy_delta,
            "reward": reward,
            "before_labels": before_labels,
            "after_labels": after_labels,
        }

    return rewards, details


def edge_semantic_losses(
    edge_log_probs: List[Dict[str, Any]],
    edge_rewards: Dict[str, float],
    semantic_lambda: float,
    sparsity_penalty: float,
    correctness_reward: float,
    edge_reward_clip: float = 5.0,
) -> List[torch.Tensor]:
    if semantic_lambda <= 0 or not edge_log_probs or correctness_reward <= 0:
        return []

    scaled_rewards = scale_edge_rewards(edge_rewards, edge_reward_clip)
    losses = []
    for edge_info in edge_log_probs:
        local_reward = semantic_lambda * scaled_rewards.get(edge_key(edge_info), 0.0)
        if local_reward != 0:
            losses.append(-edge_info["log_prob"] * float(local_reward))
    return losses


def wrong_answer_edge_losses(
    edge_log_probs: List[Dict[str, Any]],
) -> List[torch.Tensor]:
    if not edge_log_probs:
        return []
    return [
        -edge_info["log_prob"] * WRONG_EDGE_REWARD
        for edge_info in edge_log_probs
    ]


def configure_trainable_parameters(model, train_node_context: bool):
    for param in model.parameters():
        param.requires_grad = False

    trainable_modules = [
        model.edge_project,
        model.edge_gru,
        model.output_edge,
        model.embedding_node_to_edge,
    ]
    if train_node_context:
        trainable_modules.extend([
            model.task_processor,
            model.prev_nodes_aggregator,
            model.node_project,
            model.node_gru,
        ])

    params = []
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
            params.append(param)
    return params


def _candidate_roles(model) -> Tuple[List[str], List[int]]:
    roles = []
    role_ids = []
    for role in model.precomputed_embeddings.keys():
        if role in model.role_to_id:
            roles.append(role)
            role_ids.append(model.role_to_id[role])
    if not roles:
        for role_id in sorted(model.id_to_role):
            role = model.id_to_role[role_id]
            roles.append(role)
            role_ids.append(role_id)
    return roles, role_ids


def sample_graph_with_edge_trace(
    model,
    task_embedding: torch.Tensor,
    role_constraints: Dict[str, str],
    edge_epsilon: float,
    train_node_context: bool,
    ref_model=None,
    feed_previous_edge_features_to_node: bool = False,
) -> Tuple[nx.DiGraph, Dict[str, List[Any]]]:
    device = model.args.device
    task_batch = task_embedding.to(device).float().view(1, -1)
    t_proc = model.task_processor(task_batch)
    use_ref_model = ref_model is not None

    roles, role_ids = _candidate_roles(model)
    end_embedding = model.full_embedding_matrix[model.END_TOKEN].unsqueeze(0)
    candidate_embs = torch.cat(
        [model.full_embedding_matrix[role_ids], end_embedding], dim=0
    )
    end_idx = len(roles)

    min_num_nodes = model.data_statistics.get("min_num_nodes", 2)
    max_num_nodes = model.data_statistics.get("max_num_nodes", 10)
    feature_len = model.embedding_dim + model.num_nodes_to_consider * model.len_edge_vec
    has_edge_token = 1

    processed_role_embeddings = {}
    for role in roles:
        emb = model.precomputed_embeddings.get(role)
        if emb is None:
            emb = torch.zeros(model.embedding_dim)
        processed_role_embeddings[role] = torch.as_tensor(
            emb, device=device, dtype=torch.float32
        ).view(-1)

    h_node = torch.zeros(
        1, 1, model.args.hidden_size_node_level_transformer, device=device
    )
    start_input = torch.zeros(1, 1, feature_len, device=device)
    start_input[:, 0, :model.embedding_dim] = t_proc
    start_input[:, 0, model.embedding_dim + model.len_edge_vec - 2] = 1
    _, h_node = model.node_gru(model.node_project(start_input), h_node)

    if use_ref_model:
        ref_device = ref_model.args.device
        ref_task_batch = task_embedding.to(ref_device).float().view(1, -1)
        with torch.no_grad():
            ref_t_proc = ref_model.task_processor(ref_task_batch)
        ref_feature_len = (
            ref_model.embedding_dim
            + ref_model.num_nodes_to_consider * ref_model.len_edge_vec
        )
        ref_processed_role_embeddings = {}
        for role in roles:
            emb = ref_model.precomputed_embeddings.get(role)
            if emb is None:
                emb = torch.zeros(ref_model.embedding_dim)
            ref_processed_role_embeddings[role] = torch.as_tensor(
                emb, device=ref_device, dtype=torch.float32
            ).view(-1)
        ref_h_node = torch.zeros(
            1,
            1,
            ref_model.args.hidden_size_node_level_transformer,
            device=ref_device,
        )
        ref_start_input = torch.zeros(1, 1, ref_feature_len, device=ref_device)
        ref_start_input[:, 0, :ref_model.embedding_dim] = ref_t_proc
        ref_start_input[:, 0, ref_model.embedding_dim + ref_model.len_edge_vec - 2] = 1
        with torch.no_grad():
            _, ref_h_node = ref_model.node_gru(
                ref_model.node_project(ref_start_input),
                ref_h_node,
            )
        ref_node_embeddings: List[torch.Tensor] = []

    generated_roles: List[str] = []
    node_embeddings: List[torch.Tensor] = []
    edges: List[Tuple[int, int]] = []
    edge_log_probs: List[Dict[str, Any]] = []
    edge_entropies: List[torch.Tensor] = []
    edge_kls: List[torch.Tensor] = []
    previous_edge_features = torch.zeros(
        model.num_nodes_to_consider * model.len_edge_vec,
        device=device,
    )
    if use_ref_model:
        ref_previous_edge_features = torch.zeros(
            ref_model.num_nodes_to_consider * ref_model.len_edge_vec,
            device=ref_device,
        )

    proc_cand = model.role_processor(candidate_embs)
    for i in range(max_num_nodes):
        current_input = torch.zeros(1, 1, feature_len, device=device)
        if i > 0:
            prev_embs = torch.stack(node_embeddings, dim=0)
            _, h_agg = model.prev_nodes_aggregator(prev_embs.unsqueeze(0))
            hist = h_agg.squeeze(0).squeeze(0)
            gate = torch.sigmoid(torch.sum(hist * t_proc[0]) / model.embedding_dim)
            combined = (1 - gate) * hist + gate * t_proc[0]
            current_input[0, 0, :model.embedding_dim] = combined
            if feed_previous_edge_features_to_node:
                current_input[0, 0, model.embedding_dim:] = previous_edge_features

        active_out, h_node = model.node_gru(model.node_project(current_input), h_node)
        if use_ref_model:
            ref_current_input = torch.zeros(1, 1, ref_feature_len, device=ref_device)
            if i > 0:
                with torch.no_grad():
                    ref_prev_embs = torch.stack(ref_node_embeddings, dim=0)
                    _, ref_h_agg = ref_model.prev_nodes_aggregator(
                        ref_prev_embs.unsqueeze(0)
                    )
                    ref_hist = ref_h_agg.squeeze(0).squeeze(0)
                    ref_gate = torch.sigmoid(
                        torch.sum(ref_hist * ref_t_proc[0]) / ref_model.embedding_dim
                    )
                    ref_combined = (1 - ref_gate) * ref_hist + ref_gate * ref_t_proc[0]
                ref_current_input[0, 0, :ref_model.embedding_dim] = ref_combined
                if feed_previous_edge_features_to_node:
                    ref_current_input[0, 0, ref_model.embedding_dim:] = (
                        ref_previous_edge_features
                    )
            with torch.no_grad():
                ref_active_out, ref_h_node = ref_model.node_gru(
                    ref_model.node_project(ref_current_input),
                    ref_h_node,
                )

        pred_node_emb = model.output_node(active_out)
        scores = torch.matmul(pred_node_emb.squeeze(1), proc_cand.t())
        probs = F.softmax(scores, dim=-1)
        if i < min_num_nodes:
            probs[:, end_idx] = 0
        probs = probs / (probs.sum(dim=1, keepdim=True) + EPS)

        node_choice = torch.multinomial(probs, 1).item()
        if node_choice == end_idx:
            break

        role_id = role_ids[node_choice]
        role = model.id_to_role.get(role_id, roles[node_choice])
        generated_roles.append(role)
        role_emb = processed_role_embeddings.get(role)
        if role_emb is None:
            role_emb = torch.randn(model.embedding_dim, device=device)
            processed_role_embeddings[role] = role_emb
        node_embeddings.append(role_emb.clone())
        if use_ref_model:
            ref_role_emb = ref_processed_role_embeddings.get(role)
            if ref_role_emb is None:
                ref_role_emb = torch.zeros(ref_model.embedding_dim, device=ref_device)
                ref_processed_role_embeddings[role] = ref_role_emb
            ref_node_embeddings.append(ref_role_emb.clone())

        edge_context = active_out if train_node_context else active_out.detach()
        h_edge = model.embedding_node_to_edge(edge_context)
        edge_input = torch.zeros(1, 1, model.len_edge_vec, device=device)
        edge_input[:, 0, model.len_edge_vec - 2] = 1
        edge_input = model.edge_project(edge_input)
        if use_ref_model:
            with torch.no_grad():
                ref_h_edge = ref_model.embedding_node_to_edge(ref_active_out)
                ref_edge_input = torch.zeros(
                    1, 1, ref_model.len_edge_vec, device=ref_device
                )
                ref_edge_input[:, 0, ref_model.len_edge_vec - 2] = 1
                ref_edge_input = ref_model.edge_project(ref_edge_input)

        selected_for_node = []
        candidate_records = []
        sampled_edge_exists: List[int] = []
        for j in range(min(model.num_nodes_to_consider, i)):
            if j > 0:
                edge_input = model.edge_project(edge_input)
                if use_ref_model:
                    with torch.no_grad():
                        ref_edge_input = ref_model.edge_project(ref_edge_input)
            edge_out, h_edge = model.edge_gru(edge_input, h_edge)
            edge_pred = model.output_edge(edge_out).view(1, model.len_edge_vec)
            edge_prob = edge_pred[:, has_edge_token].clamp(PROB_EPS, 1 - PROB_EPS)
            if use_ref_model:
                with torch.no_grad():
                    ref_edge_out, ref_h_edge = ref_model.edge_gru(
                        ref_edge_input,
                        ref_h_edge,
                    )
                    ref_edge_pred = ref_model.output_edge(ref_edge_out).view(
                        1,
                        ref_model.len_edge_vec,
                    )
                    ref_edge_prob = ref_edge_pred[:, has_edge_token].clamp(
                        PROB_EPS,
                        1 - PROB_EPS,
                    )
                edge_kls.append(
                    bernoulli_kl(edge_prob.view(()), ref_edge_prob.to(device).view(()))
                )
            sample_prob = (1 - edge_epsilon) * edge_prob + edge_epsilon * 0.5
            sample_prob = sample_prob.clamp(PROB_EPS, 1 - PROB_EPS)
            dist = torch.distributions.Bernoulli(probs=sample_prob)
            exists = dist.sample()
            log_prob = dist.log_prob(exists).view(())
            entropy = dist.entropy().view(())
            edge_entropies.append(entropy)

            next_input = torch.zeros(1, 1, model.len_edge_vec, device=device)
            next_input[:, 0, int(exists.item())] = 1
            edge_input = next_input
            sampled_edge_exists.append(int(exists.item()))
            if use_ref_model:
                ref_next_input = torch.zeros(
                    1, 1, ref_model.len_edge_vec, device=ref_device
                )
                ref_next_input[:, 0, int(exists.item())] = 1
                ref_edge_input = ref_next_input

            src = i - j - 1
            dst = i
            candidate_records.append((src, dst))
            if int(exists.item()) == has_edge_token:
                selected_for_node.append((src, dst, log_prob))

        if i > 0 and not selected_for_node and candidate_records:
            src, dst = random.choice(candidate_records)
            selected_for_node.append((src, dst, None))
            forced_j = i - src - 1
            if forced_j < len(sampled_edge_exists):
                sampled_edge_exists[forced_j] = has_edge_token

        for src, dst, log_prob in selected_for_node:
            edges.append((src, dst))
            if log_prob is not None:
                edge_log_probs.append({
                    "type": "spatial",
                    "round": 0,
                    "source_idx": src,
                    "target_idx": dst,
                    "log_prob": log_prob,
                    "edge_key": f"spatial:0:{src}->{dst}",
                })

        previous_edge_features.zero_()
        valid_edge_slots = min(model.num_nodes_to_consider, i)
        for j in range(valid_edge_slots):
            offset = j * model.len_edge_vec
            edge_token = sampled_edge_exists[j] if j < len(sampled_edge_exists) else 0
            previous_edge_features[offset + int(edge_token)] = 1
        if use_ref_model:
            ref_previous_edge_features.zero_()
            ref_valid_edge_slots = min(ref_model.num_nodes_to_consider, i)
            for j in range(ref_valid_edge_slots):
                offset = j * ref_model.len_edge_vec
                edge_token = sampled_edge_exists[j] if j < len(sampled_edge_exists) else 0
                ref_previous_edge_features[offset + int(edge_token)] = 1

    graph = nx.DiGraph()
    for idx, role in enumerate(generated_roles):
        graph.add_node(idx, label=model.role_to_id.get(role, 0), role=role)
        graph.nodes[idx]["constraint"] = role_constraints.get(role, "")
        if role in processed_role_embeddings:
            graph.nodes[idx]["role_embedding"] = (
                processed_role_embeddings[role].detach().cpu().numpy()
            )
    for src, dst in edges:
        if src in graph.nodes and dst in graph.nodes and src != dst:
            graph.add_edge(src, dst, label=1)

    if graph.number_of_nodes() > 1:
        components = list(nx.weakly_connected_components(graph))
        if len(components) > 1:
            main = max(components, key=len)
            for comp in components:
                if comp == main:
                    continue
                src = next(iter(comp))
                dst = next(iter(main))
                if src < dst:
                    graph.add_edge(src, dst, label=1)
                else:
                    graph.add_edge(dst, src, label=1)

    graph.graph["mode"] = "ARGDesignerEdgeRL"
    graph.graph["roles"] = generated_roles
    trace = {
        "edge_log_probs": edge_log_probs,
        "edge_entropies": edge_entropies,
        "edge_kls": edge_kls,
    }
    return graph, trace


async def execute_node_with_history(node, inputs, round_idx: int, num_entropy_samples: int):
    spatial_info = node.get_spatial_info()
    temporal_info = node.get_temporal_info()
    entropy_samples = await _sample_node_outputs(
        node,
        inputs,
        spatial_info,
        temporal_info,
        max(1, num_entropy_samples),
    )
    node.outputs = _as_output_list(entropy_samples[0] if entropy_samples else "")
    if not hasattr(node, "execution_history"):
        node.execution_history = []
    node.execution_history.append({
        "round": round_idx,
        "spatial_info": spatial_info,
        "temporal_info": temporal_info,
        "entropy_samples": entropy_samples,
    })
    return node.outputs


async def execute_graph_with_history(
    graph: TestGraph,
    inputs: Dict[str, Any],
    num_rounds: int,
    num_entropy_samples: int,
    max_tries: int = 3,
    max_time: int = 600,
) -> List[Any]:
    for node in graph.nodes.values():
        node.execution_history = []

    for round_idx in range(num_rounds):
        in_degree = {
            node_id: len(node.spatial_predecessors)
            for node_id, node in graph.nodes.items()
        }
        zero_in_degree_queue = [
            node_id for node_id, degree in in_degree.items() if degree == 0
        ]
        while zero_in_degree_queue:
            current_node_id = zero_in_degree_queue.pop(0)
            tries = 0
            while tries < max_tries:
                try:
                    await asyncio.wait_for(
                        execute_node_with_history(
                            graph.nodes[current_node_id],
                            inputs,
                            round_idx,
                            num_entropy_samples,
                        ),
                        timeout=max_time,
                    )
                    break
                except Exception:
                    pass
                tries += 1
            for successor in graph.nodes[current_node_id].spatial_successors:
                if successor.id not in graph.nodes:
                    continue
                in_degree[successor.id] -= 1
                if in_degree[successor.id] == 0:
                    zero_in_degree_queue.append(successor.id)
        graph.update_memory()

    graph.connect_decision_node()
    await graph.decision_node.async_execute(inputs)
    final_answers = graph.decision_node.outputs
    if len(final_answers) == 0:
        final_answers.append("No answer of the decision node")
    return final_answers


def attach_edge_trace_to_test_graph(
    test_graph: TestGraph,
    edge_trace: List[Dict[str, Any]],
    num_rounds: int,
):
    node_list = list(test_graph.nodes.values())
    mapped = []
    for edge_info in edge_trace:
        src_idx = edge_info["source_idx"]
        dst_idx = edge_info["target_idx"]
        if src_idx >= len(node_list) or dst_idx >= len(node_list):
            continue
        for round_idx in range(num_rounds):
            src_id = node_list[src_idx].id
            dst_id = node_list[dst_idx].id
            copied = dict(edge_info)
            copied["source"] = src_id
            copied["target"] = dst_id
            copied["round"] = round_idx
            copied["edge_key"] = f"spatial:{round_idx}:{src_id}->{dst_id}"
            mapped.append(copied)
    test_graph.edge_log_probs = mapped


def infinite_loader(dataset, limit_questions: Optional[int]) -> Iterator[Any]:
    indices = list(range(len(dataset)))
    if limit_questions is not None:
        indices = indices[:limit_questions]
    while True:
        random.shuffle(indices)
        for idx in indices:
            yield dataset[idx]


def _checkpoint_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _checkpoint_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_checkpoint_safe_value(item) for item in value]
    if hasattr(value, "__dict__") and not callable(value):
        return _checkpoint_safe_value(vars(value))
    return str(value)


def _checkpoint_safe_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        raw = value
    elif hasattr(value, "__dict__"):
        raw = vars(value)
    else:
        return {}
    return {
        str(key): _checkpoint_safe_value(item)
        for key, item in raw.items()
    }


def save_rl_checkpoint(model, output_dir: str, args, name: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "data_statistics": model.data_statistics,
        "args": _checkpoint_safe_dict(model.args),
        "rl_args": _checkpoint_safe_dict(args),
    }, path)
    return path


async def evaluate_current_generator(
    model,
    dataset,
    sentence_model,
    role_constraints: Dict[str, str],
    args,
    iteration: int,
):
    model.eval()
    total = min(args.eval_batch_size, len(dataset))
    if total <= 0:
        print("[eval] skipped: empty dataset")
        model.train()
        return

    start_cost = Cost.instance().value
    start_prompt_tokens = PromptTokens.instance().value
    start_completion_tokens = CompletionTokens.instance().value

    correct_count = 0
    edge_counts = []
    start_ts = time.time()

    with torch.no_grad():
        for idx in range(total):
            record = dataset[idx]
            input_dict = dataset.record_to_input(record)
            task_text = input_dict["task"]
            correct_answer = dataset.record_to_target_answer(record)
            task_embedding = torch.tensor(
                sentence_model.encode(task_text),
                device=model.args.device,
                dtype=torch.float32,
            )

            generated_graph, _ = sample_graph_with_edge_trace(
                model,
                task_embedding,
                role_constraints,
                edge_epsilon=args.eval_edge_epsilon,
                train_node_context=False,
                feed_previous_edge_features_to_node=args.feed_previous_edge_features_to_node,
            )
            edge_counts.append(generated_graph.number_of_edges())
            pyg_graph = convert_to_pyg_graph(generated_graph, task_text)
            test_graph = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=pyg_graph,
            )

            try:
                raw_answer = await test_graph.arun(input_dict, args.num_rounds)
                answer = dataset.postprocess_answer(raw_answer)
            except Exception:
                answer = ""

            if answer == correct_answer:
                correct_count += 1

    eval_cost = Cost.instance().value - start_cost
    eval_prompt_tokens = PromptTokens.instance().value - start_prompt_tokens
    eval_completion_tokens = CompletionTokens.instance().value - start_completion_tokens
    accuracy = correct_count / max(1, total)
    print(
        f"[eval iter {iteration}] "
        f"accuracy={accuracy:.3f} ({correct_count}/{total}) "
        f"cost=${eval_cost:.6f} "
        f"prompt_tokens={int(eval_prompt_tokens)} "
        f"completion_tokens={int(eval_completion_tokens)}"
    )
    model.train()


async def train_rl(args):
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
    dataset = MMLUDataset("dev")
    eval_dataset = MMLUDataset(args.eval_split) if args.eval_every > 0 else None
    loader = infinite_loader(dataset, args.limit_questions)
    role_constraints = ROLE_DESCRIPTION

    num_entropy_samples = max(2, int(args.num_entropy_samples))
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
            "OPENAI_API_KEY, or --semantic_judge_base_url for a local OpenAI-compatible server."
        )

    metrics_path = os.path.join(args.output_dir, "rl_metrics.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    for iteration in range(args.num_iterations):
        start_ts = time.time()
        losses: List[torch.Tensor] = []
        correctness_values = []
        edge_counts = []
        reward_summaries = []
        answers = []
        kl_values = []

        records = [next(loader) for _ in range(args.batch_size)]
        samples_per_prompt = max(1, int(args.samples_per_prompt))
        for batch_idx, record in enumerate(records):
            input_dict = dataset.record_to_input(record)
            task_text = input_dict["task"]
            correct_answer = dataset.record_to_target_answer(record)
            task_embedding = torch.tensor(
                sentence_model.encode(task_text),
                device=model.args.device,
                dtype=torch.float32,
            )

            for sample_idx in range(samples_per_prompt):
                generated_graph, trace = sample_graph_with_edge_trace(
                    model,
                    task_embedding,
                    role_constraints,
                    edge_epsilon=args.edge_epsilon,
                    train_node_context=args.train_node_context,
                    ref_model=ref_model,
                    feed_previous_edge_features_to_node=args.feed_previous_edge_features_to_node,
                )
                pyg_graph = convert_to_pyg_graph(generated_graph, task_text)
                test_graph = TestGraph(
                    domain=args.domain,
                    llm_name=args.llm_name,
                    decision_method=args.decision_method,
                    pyg_data=pyg_graph,
                )
                attach_edge_trace_to_test_graph(
                    test_graph,
                    trace["edge_log_probs"],
                    args.num_rounds,
                )

                try:
                    raw_answer = await execute_graph_with_history(
                        test_graph,
                        input_dict,
                        args.num_rounds,
                        num_entropy_samples,
                    )
                    answer = dataset.postprocess_answer(raw_answer)
                except Exception:
                    answer = ""

                is_correct = answer == correct_answer
                correctness = 1.0 if is_correct else 0.0
                correctness_values.append(correctness)
                edge_counts.append(generated_graph.number_of_edges())
                answers.append(answer)

                edge_rewards: Dict[str, float] = {}
                edge_details: Dict[str, Dict[str, Any]] = {}
                if is_correct:
                    edge_rewards, edge_details = await edge_entropy_rewards(
                        test_graph,
                        task_text,
                        input_dict,
                        judge,
                        num_entropy_samples,
                        negative_reward_scale=args.negative_edge_reward_scale,
                        nonpositive_penalty=args.nonpositive_edge_penalty,
                    )

                sample_loss = torch.tensor(0.0, device=model.args.device)
                if is_correct:
                    edge_losses = edge_semantic_losses(
                        test_graph.edge_log_probs,
                        edge_rewards,
                        semantic_lambda=args.semantic_lambda,
                        sparsity_penalty=args.sparsity_penalty,
                        correctness_reward=correctness,
                        edge_reward_clip=args.edge_reward_clip,
                    )
                    if edge_losses:
                        sample_loss = torch.stack(edge_losses).sum()
                else:
                    edge_losses = wrong_answer_edge_losses(test_graph.edge_log_probs)
                    if edge_losses:
                        sample_loss = torch.stack(edge_losses).mean()

                if is_correct:
                    if trace["edge_entropies"]:
                        entropy_bonus = torch.stack(trace["edge_entropies"]).mean()
                        sample_loss = sample_loss - args.entropy_coef * entropy_bonus

                    if args.kl_coef > 0 and trace.get("edge_kls"):
                        kl_loss = torch.stack(trace["edge_kls"]).mean()
                        sample_loss = sample_loss + args.kl_coef * kl_loss
                        kl_values.append(float(kl_loss.detach().cpu()))

                if sample_loss.requires_grad:
                    losses.append(sample_loss)

                reward_summaries.append({
                    "task": task_text,
                    "sample_idx": sample_idx,
                    "correct": is_correct,
                    "edge_rewards": edge_rewards,
                    "scaled_edge_rewards": scale_edge_rewards(
                        edge_rewards,
                        args.edge_reward_clip,
                    ),
                    "edge_details": edge_details,
                    "num_edges": generated_graph.number_of_edges(),
                })
        if losses:
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            loss_value = float(total_loss.detach().cpu())
        else:
            loss_value = 0.0

        correct_rate = sum(correctness_values) / max(1, len(correctness_values))
        avg_edges = sum(edge_counts) / max(1, len(edge_counts))
        metric = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": iteration + 1,
            "loss": loss_value,
            "correct_rate": correct_rate,
            "avg_edges": avg_edges,
            "avg_kl": sum(kl_values) / max(1, len(kl_values)),
            "samples_per_prompt": samples_per_prompt,
            "answers": answers,
            "reward_summaries": reward_summaries,
            "cost": Cost.instance().value,
            "prompt_tokens": PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value,
            "elapsed_sec": time.time() - start_ts,
        }
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(metric, ensure_ascii=False, default=str) + "\n")

        print(
            f"Iter {iteration + 1}/{args.num_iterations}: "
            f"loss={loss_value:.4f} correct_rate={correct_rate:.3f} "
            f"avg_edges={avg_edges:.2f} "
            f"avg_kl={sum(kl_values) / max(1, len(kl_values)):.6f} "
            f"time={time.time() - start_ts:.1f}s"
        )

        if args.eval_every > 0 and (iteration + 1) % args.eval_every == 0:
            await evaluate_current_generator(
                model,
                eval_dataset,
                sentence_model,
                role_constraints,
                args,
                iteration + 1,
            )

    save_rl_checkpoint(model, args.output_dir, args, RL_FINAL_CHECKPOINT)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(train_rl(parse_args()))
