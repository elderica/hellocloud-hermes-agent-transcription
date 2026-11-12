import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import __init__ as plugin


class HelloCloudTranscriptionProviderTest(unittest.TestCase):
    def setUp(self):
        self.audio = tempfile.NamedTemporaryFile(suffix=".mp3")
        self.audio.write(b"test-audio")
        self.audio.flush()

    def tearDown(self):
        self.audio.close()

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_transcribes_with_default_model_and_normalizes_response(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="  こんにちは、世界。  ",
            duration=3.4,
            language="ja",
            usage=SimpleNamespace(seconds=4),
        )

        result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertEqual(
            result,
            {
                "success": True,
                "transcript": "こんにちは、世界。",
                "provider": "hellocloud",
                "model": "whisper-large-v3-turbo",
                "duration": 3.4,
                "language": "ja",
                "usage_seconds": 4,
            },
        )
        request = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(request["model"], "whisper-large-v3-turbo")
        self.assertEqual(request["response_format"], "verbose_json")
        self.assertNotIn("language", request)
        self.assertNotIn("prompt", request)
        self.assertEqual(request["file"].name, self.audio.name)
        openai_class.assert_called_once_with(
            api_key="hello_test",
            base_url="https://api.router.hello.ai/v1",
            timeout=300.0,
            max_retries=1,
        )
        client.close.assert_called_once_with()

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_forwards_model_language_and_prompt_overrides(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="文字起こし",
            duration=2.0,
            language="ja",
            usage=SimpleNamespace(seconds=2),
        )

        result = plugin.HelloCloudTranscriptionProvider().transcribe(
            self.audio.name,
            model="whisper-large-v3",
            language="ja",
            prompt="固有名詞: HelloCloud",
            ignored_option="ignored",
        )

        self.assertTrue(result["success"])
        request = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(request["model"], "whisper-large-v3")
        self.assertEqual(request["language"], "ja")
        self.assertEqual(request["prompt"], "固有名詞: HelloCloud")
        self.assertNotIn("ignored_option", request)

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_api_failure_returns_error_envelope_and_closes_client(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        client.audio.transcriptions.create.side_effect = RuntimeError("upstream unavailable")

        result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertEqual(
            result,
            {
                "success": False,
                "transcript": "",
                "provider": "hellocloud",
                "error": "HelloCloud 文字起こしに失敗しました: upstream unavailable",
            },
        )
        client.close.assert_called_once_with()

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_empty_transcript_returns_no_speech(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="   ", duration=1.0, language="ja", usage=SimpleNamespace(seconds=1)
        )

        result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertEqual(
            result,
            {
                "success": False,
                "transcript": "",
                "provider": "hellocloud",
                "error": "HelloCloud が空の文字起こし結果を返しました",
                "no_speech": True,
            },
        )

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="")
    def test_missing_credentials_fail_before_creating_client(
        self, _resolve_api_key, openai_class
    ):
        result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertFalse(result["success"])
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["provider"], "hellocloud")
        self.assertIn("HELLOCLOUD_API_KEY", result["error"])
        self.assertIn("有料キー", result["error"])
        openai_class.assert_not_called()

    def test_model_catalog_contains_both_hellocloud_whisper_models(self):
        provider = plugin.HelloCloudTranscriptionProvider()

        models = provider.list_models()

        self.assertEqual(provider.default_model(), "whisper-large-v3-turbo")
        self.assertEqual(
            [model["id"] for model in models],
            ["whisper-large-v3-turbo", "whisper-large-v3"],
        )
        self.assertEqual(models[0]["price"], "¥0.025/秒 (¥1.5/分)")
        self.assertEqual(models[1]["price"], "¥0.03/秒 (¥1.8/分)")
        self.assertTrue(all(model["max_file_bytes"] == 25 * 1024 * 1024 for model in models))

    def test_availability_requires_api_key_and_openai_sdk(self):
        provider = plugin.HelloCloudTranscriptionProvider()

        with (
            patch.object(plugin, "_resolve_api_key", return_value="hello_test"),
            patch("importlib.util.find_spec", return_value=object()),
        ):
            self.assertTrue(provider.is_available())
        with patch.object(plugin, "_resolve_api_key", return_value=""):
            self.assertFalse(provider.is_available())
        with (
            patch.object(plugin, "_resolve_api_key", return_value="hello_test"),
            patch("importlib.util.find_spec", return_value=None),
        ):
            self.assertFalse(provider.is_available())
        with patch.object(plugin, "_resolve_api_key", side_effect=RuntimeError("no secret scope")):
            self.assertFalse(provider.is_available())

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_provider_config_controls_model_endpoint_transport_and_temperature(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="設定付き", duration=2.0, language="ja", usage=SimpleNamespace(seconds=2)
        )
        config = {
            "stt": {
                "hellocloud": {
                    "model": "whisper-large-v3",
                    "base_url": "https://example.test/custom/v1/",
                    "timeout": 42,
                    "max_retries": 3,
                    "temperature": 0.2,
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config):
            result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "whisper-large-v3")
        openai_class.assert_called_once_with(
            api_key="hello_test",
            base_url="https://example.test/custom/v1",
            timeout=42.0,
            max_retries=3,
        )
        request = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(request["model"], "whisper-large-v3")
        self.assertEqual(request["temperature"], 0.2)

    @patch("openai.OpenAI", side_effect=RuntimeError("SDK setup failed"))
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_client_setup_failure_returns_error_envelope(
        self, _resolve_api_key, _openai_class
    ):
        result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertEqual(
            result,
            {
                "success": False,
                "transcript": "",
                "provider": "hellocloud",
                "error": "HelloCloud 文字起こしに失敗しました: SDK setup failed",
            },
        )

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_missing_response_metadata_does_not_discard_transcript(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="本文だけ")

        result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

        self.assertEqual(
            result,
            {
                "success": True,
                "transcript": "本文だけ",
                "provider": "hellocloud",
                "model": "whisper-large-v3-turbo",
            },
        )

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_unknown_model_is_rejected_before_api_call(
        self, _resolve_api_key, openai_class
    ):
        result = plugin.HelloCloudTranscriptionProvider().transcribe(
            self.audio.name, model="whisper-1"
        )

        self.assertEqual(
            result,
            {
                "success": False,
                "transcript": "",
                "provider": "hellocloud",
                "error": "未対応のモデルです: whisper-1 (whisper-large-v3-turbo, whisper-large-v3 から選択)",
            },
        )
        openai_class.assert_not_called()

    @patch("openai.OpenAI")
    @patch.object(plugin, "_resolve_api_key", return_value="hello_test")
    def test_invalid_temperature_is_rejected_before_api_call(
        self, _resolve_api_key, openai_class
    ):
        client = openai_class.return_value
        for value in (-0.1, 1.1, "hot"):
            with self.subTest(value=value), patch(
                "hermes_cli.config.load_config",
                return_value={"stt": {"hellocloud": {"temperature": value}}},
            ):
                result = plugin.HelloCloudTranscriptionProvider().transcribe(self.audio.name)

            self.assertFalse(result["success"])
            self.assertIn("temperature", result["error"])
            self.assertIn("0〜1", result["error"])
        client.audio.transcriptions.create.assert_not_called()

    def test_setup_schema_identifies_paid_hellocloud_provider(self):
        schema = plugin.HelloCloudTranscriptionProvider().get_setup_schema()

        self.assertEqual(schema["name"], "HelloCloud")
        self.assertEqual(schema["badge"], "paid")
        self.assertIn("HELLOCLOUD_API_KEY", schema["tag"])
        self.assertEqual(schema["env_vars"], [])


if __name__ == "__main__":
    unittest.main()
