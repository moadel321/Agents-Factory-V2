from typing import Any, Dict, Optional


class SendSMSTask:
    def __init__(self, chat_ctx):
        self.chat_ctx = chat_ctx

    async def run(self, to: str, body: str, timeout_ms: int = 10000, retries: int = 0) -> Dict[str, Any]:
        return {"sent": True, "to": to}


class TransferCallTask:
    def __init__(self, chat_ctx, job_context):
        self.chat_ctx = chat_ctx
        self.job_context = job_context

    async def run(self, phone_number: str, timeout_ms: int = 10000, retries: int = 0) -> Dict[str, Any]:
        return {"transferred": True, "to": phone_number}


class RestWebhookTask:
    def __init__(self, chat_ctx):
        self.chat_ctx = chat_ctx

    async def run(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, body: Optional[Dict[str, Any]] = None, timeout_ms: int = 10000, retries: int = 0) -> Dict[str, Any]:
        return {"ok": True, "url": url}


