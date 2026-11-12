"""HelloCloud speech-to-text provider for Hermes Agent."""

from __future__ import annotations

import importlib.util
from typing import Any, Optional

from agent.secret_scope import get_secret_str
from agent.transcription_provider import TranscriptionProvider

API_KEY_ENV = "HELLOCLOUD_API_KEY"
DEFAULT_BASE_URL = "https://api.router.hello.ai/v1"
DEFAULT_MODEL = "whisper-large-v3-turbo"
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_FORMATS = ["mp3", "mp4", "m4a", "wav", "flac", "ogg", "webm"]


def _resolve_api_key() -> str:
    return get_secret_str(API_KEY_ENV).strip()


def _provider_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        stt = config.get("stt") if isinstance(config, dict) else {}
        section = stt.get("hellocloud") if isinstance(stt, dict) else {}
        return section if isinstance(section, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


class HelloCloudTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "hellocloud"

    @property
    def display_name(self) -> str:
        return "HelloCloud"

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "whisper-large-v3-turbo",
                "display": "Whisper Large V3 Turbo (HelloCloud)",
                "price": "¥0.025/秒 (¥1.5/分)",
                "max_file_bytes": MAX_FILE_BYTES,
                "supported_formats": list(SUPPORTED_FORMATS),
            },
            {
                "id": "whisper-large-v3",
                "display": "Whisper Large V3 (HelloCloud)",
                "price": "¥0.03/秒 (¥1.8/分)",
                "max_file_bytes": MAX_FILE_BYTES,
                "supported_formats": list(SUPPORTED_FORMATS),
            },
        ]

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def is_available(self) -> bool:
        try:
            return bool(_resolve_api_key()) and importlib.util.find_spec("openai") is not None
        except Exception:  # noqa: BLE001
            return False

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "paid",
            "tag": "Whisper Large V3 / Turbo via HelloCloud — HELLOCLOUD_API_KEY (環境変数)",
            "env_vars": [],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        client = None
        try:
            api_key = _resolve_api_key()
            if not api_key:
                return {
                    "success": False,
                    "transcript": "",
                    "provider": self.name,
                    "error": "HELLOCLOUD_API_KEY が設定されていません。有料キー (hello_...) を設定してください。",
                }
            import openai

            config = _provider_config()
            model_id = (model or "").strip() or str(config.get("model") or DEFAULT_MODEL).strip()
            supported_models = {entry["id"] for entry in self.list_models()}
            if model_id not in supported_models:
                choices = ", ".join(entry["id"] for entry in self.list_models())
                return {
                    "success": False,
                    "transcript": "",
                    "provider": self.name,
                    "error": f"未対応のモデルです: {model_id} ({choices} から選択)",
                }
            base_url = str(config.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")
            temperature = config.get("temperature")
            if temperature is not None:
                try:
                    temperature = float(temperature)
                    if not 0 <= temperature <= 1:
                        raise ValueError
                except (TypeError, ValueError):
                    return {
                        "success": False,
                        "transcript": "",
                        "provider": self.name,
                        "error": "stt.hellocloud.temperature は 0〜1 の数値で指定してください",
                    }
            try:
                timeout = float(config.get("timeout", REQUEST_TIMEOUT_SECONDS))
            except (TypeError, ValueError):
                timeout = REQUEST_TIMEOUT_SECONDS
            try:
                max_retries = int(config.get("max_retries", 1))
            except (TypeError, ValueError):
                max_retries = 1
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            request: dict[str, Any] = {
                "model": model_id,
                "response_format": "verbose_json",
            }
            if language:
                request["language"] = language
            prompt = extra.get("prompt")
            if prompt:
                request["prompt"] = prompt
            if temperature is not None:
                request["temperature"] = temperature
            with open(file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(file=audio_file, **request)
            usage = getattr(response, "usage", None)
            transcript = response.text.strip()
            if not transcript:
                return {
                    "success": False,
                    "transcript": "",
                    "provider": self.name,
                    "error": "HelloCloud が空の文字起こし結果を返しました",
                    "no_speech": True,
                }
            result = {
                "success": True,
                "transcript": transcript,
                "provider": self.name,
                "model": model_id,
            }
            duration = getattr(response, "duration", None)
            if duration is not None:
                result["duration"] = duration
            response_language = getattr(response, "language", None)
            if response_language:
                result["language"] = response_language
            usage_seconds = getattr(usage, "seconds", None)
            if usage_seconds is not None:
                result["usage_seconds"] = usage_seconds
            return result
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": f"HelloCloud 文字起こしに失敗しました: {exc}",
            }
        finally:
            if client is not None:
                client.close()


def register(ctx) -> None:
    ctx.register_transcription_provider(HelloCloudTranscriptionProvider())
