"""
LLM clients (cloud-only).
- GroqChat: used for grounded QA & explanations (streamed).
- make_mistral_endpoint: HF Inference API (conversational) → ChatHuggingFace.
"""

from typing import List, Any, Dict, Optional
from pydantic import PrivateAttr
from groq import Groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from app.config.settings import Config


class GroqChat(BaseChatModel):
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 1024
    _client: Groq = PrivateAttr(default_factory=lambda: Groq(api_key=Config.GROQ_API_KEY))

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> AIMessage:
        payload = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
        ]
        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=payload,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            top_p=1,
            stream=True,
            stop=stop,
        )
        out: List[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                out.append(delta)
        return AIMessage(content="".join(out))

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        msg = self._call(messages, stop=stop)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str: return "groq_chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature, "max_tokens": self.max_tokens}


def make_mistral_endpoint(model_id: str, max_new_tokens: int = 128, temperature: float = 0.1) -> ChatHuggingFace:
    """
    HF Inference API (conversational) → ChatHuggingFace.
    Keeps your existing factory name & call sites intact.
    """
    client = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",          # <-- important (model exposes conversational provider)
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.05,
        huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_TOKEN,
    )
    return ChatHuggingFace(llm=client)
