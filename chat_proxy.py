# chat_proxy.py
import os
import requests
from typing import List
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage


class ChatAIProxy(SimpleChatModel):
    model_name: str = "llama3-70b-8192"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "chat-groq"

    def _call(self, messages: List[BaseMessage], **kwargs) -> str:
        api_key = os.getenv("AIPROXY_API_KEY")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": [
                    {"role": self._convert_role(m), "content": m.content}
                    for m in messages
                ],
            },
            timeout=30,
        )

        data = response.json()
        print("LLM response:", response.status_code, data)

        if response.status_code != 200:
            raise RuntimeError(f"LLM Error {response.status_code}: {data.get('error', data)}")

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Empty or malformed LLM response: {data}")

        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        if finish_reason == "tool_calls":
            raise RuntimeError("Groq returned a tool_call, which this agent doesn't currently support.")

        content = message.get("content")
        if not content:
            raise RuntimeError(f"Groq response had no content. Full response: {data}")

        return content


    
    def _convert_role(self, message: BaseMessage) -> str:
        if message.type == "human":
            return "user"
        elif message.type == "ai":
            return "assistant"
        else:
            return message.type

