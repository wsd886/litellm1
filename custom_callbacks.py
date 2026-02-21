import json
import os
from pathlib import Path
from threading import Lock
from typing import Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


class CaptureClaudeCodeOnlyHandler(CustomLogger):
    _file_write_lock = Lock()

    def __init__(self):
        super().__init__()
        self.output_path = os.getenv("CC_CAPTURE_FILE", "/opt/cc_requests.jsonl")
        self._write_failed_once = False

    def _append_record_to_file(self, record: dict) -> None:
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self._file_write_lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @staticmethod
    def _normalize_headers(headers: dict) -> dict:
        if not isinstance(headers, dict):
            return {}
        return {str(k).lower(): v for k, v in headers.items()}

    @staticmethod
    def _is_claude_code_request(
        headers: dict,
        body: dict,
        url: Optional[str],
        call_type: CallTypesLiteral,
    ) -> bool:
        normalized_headers = CaptureClaudeCodeOnlyHandler._normalize_headers(headers)
        user_agent = str(normalized_headers.get("user-agent", "")).lower()
        _ = body
        _ = url
        _ = call_type

        # Strict filter: only capture traffic that advertises Claude Code-like user agents.
        claude_code_ua_markers = (
            "claude-code",
            "claude code",
            "claude-cli",
            "anthropic-claude-code",
        )
        return any(marker in user_agent for marker in claude_code_ua_markers)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ):
        proxy_server_request = data.get("proxy_server_request", {}) or {}
        headers = proxy_server_request.get("headers", {}) or {}
        body = proxy_server_request.get("body", {}) or {}
        url = proxy_server_request.get("url")

        if not self._is_claude_code_request(
            headers=headers,
            body=body,
            url=url,
            call_type=call_type,
        ):
            return data

        record = {
            "header": headers,
            "body": body,
        }
        try:
            self._append_record_to_file(record=record)
        except Exception as e:
            # Fallback to container logs if writing to file fails.
            if not self._write_failed_once:
                self._write_failed_once = True
                print(
                    f"CC_CAPTURE_FILE_WRITE_ERROR path={self.output_path} error={str(e)}",
                    flush=True,
                )
            print("CC_CAPTURE " + json.dumps(record, ensure_ascii=False), flush=True)
        return data


proxy_handler_instance = CaptureClaudeCodeOnlyHandler()
