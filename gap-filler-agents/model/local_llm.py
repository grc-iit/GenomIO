

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Any, Dict
import requests
from pydantic import Field


class ChatLlamaCpp(BaseChatModel):
    base_url: str = Field(default="http://localhost:8000")
    model_name: str = Field(default="llama-2-7b-model")

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        return [
            {
                "role": (
                    "system" if isinstance(m, SystemMessage)
                    else "user" if isinstance(m, HumanMessage)
                    else "assistant"
                ),
                "content": m.content,
            }
            for m in messages
        ]

    def _generate(self, messages: List[BaseMessage], stop=None, **kwargs: Any) -> ChatResult:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": 0.5,
            "tools": kwargs.get("tools", []), 
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        print("Raw LLM response:", data)

        choice = data["choices"][0]

        if "tool_calls" in choice["message"]:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=choice["message"].get("content", ""),
                            additional_kwargs={"tool_calls": choice["message"]["tool_calls"]}
                        )
                    )
                ]
            )
        else:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=choice["message"]["content"]))]
            )

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"base_url": self.base_url, "model_name": self.model_name}
