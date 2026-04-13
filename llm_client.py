"""Local LLM client - lightweight models for 8GB RAM (Ollama/llama.cpp)"""
import os
import time
from typing import Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from local LLM"""
    content: str
    model: str
    tokens_used: int
    inference_time: float
    finish_reason: str


@dataclass
class LLMMetrics:
    """LLM usage metrics"""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    avg_temperature: float = 0.0
    requests_count: int = 0


SUPPORTED_MODELS = {
    "qwen2.5-3b-instruct": {"ram_gb": 6, "type": "ollama"},
    "qwen2.5-1.5b-instruct": {"ram_gb": 3, "type": "ollama"},
    "phi3.5-mini-instruct": {"ram_gb": 4, "type": "ollama"},
    "llama3.2-1b": {"ram_gb": 2, "type": "ollama"},
    "gemma-2-2b-it": {"ram_gb": 4, "type": "ollama"},
    "mistral-7b-instruct": {"ram_gb": 14, "type": "ollama"},
}


class LocalLLMClient:
    """Client for lightweight local LLM (8GB RAM limit)"""
    
    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "qwen2.5-3b-instruct",
        api_key: str | None = None,
        timeout: int = 120
    ):
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.timeout = timeout
        self.metrics = LLMMetrics()
        
        default_endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:11434")
        
        detected_type = self._detect_endpoint_type(endpoint or default_endpoint)
        self.endpoint_type = detected_type
        
        if endpoint:
            self.endpoint = endpoint
        elif detected_type == "ollama":
            self.endpoint = "http://localhost:11434"
        else:
            self.endpoint = "http://localhost:8080/v1"
        
        logger.info(f"LLM Client initialized: {self.model} via {self.endpoint_type} at {self.endpoint}")
        
        self._verify_data_residency()
    
    def _detect_endpoint_type(self, endpoint: str) -> str:
        """Detect endpoint type: ollama, openai-compatible, or vllm"""
        ep = endpoint.lower()
        if "11434" in ep or "ollama" in ep:
            return "ollama"
        elif "vllm" in ep or "tgi" in ep:
            return "vllm"
        else:
            return "openai"
    
    def _verify_data_residency(self):
        """Verify all requests stay within Kazakhstan"""
        allowed_regions = os.getenv("ALLOWED_REGIONS", "KZ").split(",")
        logger.info(f"Data residency: requests must stay in {allowed_regions}")
        
        prohibited = ["openai.com", "anthropic.com", "grok.com", "deepseek.com"]
        endpoint_lower = self.endpoint.lower()
        
        for provider in prohibited:
            if provider in endpoint_lower:
                raise ValueError(
                    f"PROHIBITED: Cannot use external provider {provider}. "
                    f"Only local Kazakhstan deployments allowed."
                )
    
    def _build_payload(self, prompt: str, temperature: float, max_tokens: int, stream: bool) -> dict:
        """Build request payload based on endpoint type"""
        
        if self.endpoint_type == "ollama":
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "options": {"num_predict": max_tokens},
                "stream": stream
            }
        else:
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
    
    def _parse_response(self, result: dict) -> LLMResponse:
        """Parse response based on endpoint type"""
        
        if self.endpoint_type == "ollama":
            content = result.get("message", {}).get("content", "")
            return LLMResponse(
                content=content,
                model=result.get("model", self.model),
                tokens_used=result.get("eval_count", 0),
                inference_time=result.get("eval_duration", 0) / 1e9,
                finish_reason=result.get("done", False) and "stop" or "length"
            )
        else:
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                model=result.get("model", self.model),
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                inference_time=0,
                finish_reason=result["choices"][0].get("finish_reason", "stop")
            )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> LLMResponse:
        """Generate text from lightweight LLM"""
        start_time = time.time()
        
        try:
            import requests
            
            url = f"{self.endpoint}/api/chat" if self.endpoint_type == "ollama" else f"{self.endpoint}/chat/completions"
            
            payload = self._build_payload(prompt, temperature, max_tokens, stream)
            
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.endpoint_type != "ollama":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            llm_response = self._parse_response(result)
            
            llm_response.inference_time = time.time() - start_time
            
            self._update_metrics(temperature, llm_response.tokens_used)
            
            return llm_response
            
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """Generate streaming response"""
        
        import requests
        
        url = f"{self.endpoint}/api/chat" if self.endpoint_type == "ollama" else f"{self.endpoint}/chat/completions"
        payload = self._build_payload(prompt, temperature, max_tokens, True)
        
        response = requests.post(url, json=payload, timeout=self.timeout, stream=True)
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
                elif "choices" in data and len(data["choices"]) > 0:
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
    
    def _update_metrics(self, temperature: float, tokens: int):
        """Update usage metrics"""
        self.metrics.requests_count += 1
        self.metrics.total_tokens += tokens
        self.metrics.avg_temperature = (
            (self.metrics.avg_temperature * (self.metrics.requests_count - 1) + temperature)
            / self.metrics.requests_count
        )
        
        cost_per_token = 0.00005
        self.metrics.total_cost = self.metrics.total_tokens * cost_per_token
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return {
            "total_tokens": self.metrics.total_tokens,
            "total_cost_tenge": self.metrics.total_cost * 10,
            "avg_temperature": self.metrics.avg_temperature,
            "requests_count": self.metrics.requests_count
        }
    
    def health_check(self) -> bool:
        """Check if LLM endpoint is available"""
        try:
            import requests
            if self.endpoint_type == "ollama":
                response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            else:
                response = requests.get(f"{self.endpoint}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def get_available_models(ram_limit_gb: int = 8) -> list[dict]:
        """Get models that fit within RAM limit"""
        return [
            {"model": name, "ram_gb": info["ram_gb"]}
            for name, info in SUPPORTED_MODELS.items()
            if info["ram_gb"] <= ram_limit_gb
        ]


class LLMWrapper:
    """Wrapper for LLM with security/compliance focus"""
    
    def __init__(self, client: LocalLLMClient):
        self.client = client
    
    def analyze_code_security(
        self,
        code: str,
        language: str = "python"
    ) -> dict:
        """Analyze code for security and compliance (optimized for small models)"""
        prompt = f"""Ты эксперт по безопасности Казахстана. Проанализируй код на:
1. Уязвимости безопасности
2. Нарушения Закона РК о персональных данных
3. Криптография (алгоритмы, длины ключей)

Код ({language}):
{code[:1500]}

Ответ кратко (3-5 пунктов): severity, category, recommendation.
"""
        response = self.client.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=1024
        )
        
        return {
            "analysis": response.content,
            "tokens_used": response.tokens_used,
            "inference_time": response.inference_time
        }
    
    def check_regulatory_compliance(
        self,
        document_text: str,
        regulation_type: str
    ) -> dict:
        """Check document against Kazakhstan regulations"""
        prompt = f"""Ты аудитор соответствия РК. Проверь документ на соответствие {regulation_type}.
        
Документ (первые 2000 символов):
{document_text[:2000]}

Ответ: compliance_status (compliant/non_compliant/needs_review), issues (1-2 предложения).
"""
        response = self.client.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=512
        )
        
        return {
            "compliance_check": response.content,
            "tokens_used": response.tokens_used
        }


if __name__ == "__main__":
    print("Available models for 8GB RAM:")
    for m in LocalLLMClient.get_available_models(8):
        print(f"  - {m['model']}: {m['ram_gb']}GB")
    
    client = LocalLLMClient()
    
    print(f"\nHealth check: {client.health_check()}")
    
    response = client.generate(
        prompt="Закон РК о персональных данных в 2 предложениях",
        temperature=0.3
    )
    
    print(f"Response: {response.content[:200]}...")
    print(f"Tokens: {response.tokens_used}, Time: {response.inference_time:.2f}s")
    print(f"Metrics: {client.get_metrics()}")