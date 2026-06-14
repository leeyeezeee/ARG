import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from mas_framework.llm.format import Message
from mas_framework.llm.price import cost_count, cost_count_from_usage
from mas_framework.llm.llm import LLM
from mas_framework.llm.llm_registry import LLMRegistry


load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEY = os.getenv('API_KEY')
from openai import OpenAI, AsyncOpenAI


def _message_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _messages_to_prompt_text(messages: List[Dict]) -> str:
    text_parts = []
    for message in messages:
        if hasattr(message, "dict"):
            message = message.dict()
        elif not isinstance(message, dict):
            message = {
                "role": getattr(message, "role", "user"),
                "content": getattr(message, "content", str(message)),
            }
        role = message.get("role", "")
        content = _message_content_to_text(message.get("content", ""))
        text_parts.append(f"{role}: {content}")
    return "\n".join(text_parts)


def _usage_value(usage, name: str, default: int = 0) -> int:
    if usage is None:
        return default
    if isinstance(usage, dict):
        return int(usage.get(name, default) or default)
    return int(getattr(usage, name, default) or default)


def _record_usage_or_estimate(model: str, messages: List[Dict], response: str, chat_completion) -> None:
    usage = getattr(chat_completion, "usage", None)
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    if prompt_tokens or completion_tokens:
        cost_count_from_usage(prompt_tokens, completion_tokens, model)
        return

    prompt_text = _messages_to_prompt_text(messages)
    cost_count(prompt_text, response or "", model)


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
        model: str,
        msg: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        num_comps: Optional[int] = 1,
):
    client = AsyncOpenAI(base_url=MINE_BASE_URL, api_key=MINE_API_KEY)
    
    # Disable Qwen thinking mode for local OpenAI-compatible servers.
    if "qwen" in model.lower():
        chat_completion = await client.chat.completions.create(
            messages=msg,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
    else:
        chat_completion = await client.chat.completions.create(
            messages=msg,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    response = chat_completion.choices[0].message.content
    _record_usage_or_estimate(model, msg, response, chat_completion)
    return response


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name, messages)

    def gen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
