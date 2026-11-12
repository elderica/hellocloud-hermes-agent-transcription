# hellocloud-hermes-agent-transcription

[Hermes Agent](https://hermes-agent.nousresearch.com/) 用の **HelloCloud 音声文字起こしバックエンドプラグイン** です。
[HelloCloud](https://router.hello.ai) の OpenAI 互換 API (`/v1/audio/transcriptions`) 経由で、
Whisper Large V3 / Whisper Large V3 Turbo を利用できます。

- Hermes の公開 `TranscriptionProvider` API に準拠
- 音声メッセージとボイスモードの文字起こしを `stt.provider: hellocloud` で処理
- `language`・`prompt`・`temperature` に対応
- `verbose_json` から再生時間・言語・課金秒数を結果に付加
- 25 MB を超える音声は Hermes 本体が再エンコード・分割
- API キーは **環境変数 `HELLOCLOUD_API_KEY`** から取得

API 仕様: [HelloCloud 音声文字起こしドキュメント](https://router.hello.ai/docs/transcriptions/)

## 対応モデル

| モデル ID | 精度 | 料金 |
|-----------|------|------|
| `whisper-large-v3-turbo` (既定) | FP16 | ¥0.025 / 秒 (¥1.5 / 分) |
| `whisper-large-v3` | FP16 | ¥0.03 / 秒 (¥1.8 / 分) |

課金は音声の長さで、1 秒未満は切り上げ、最低 1 秒です。定額プランではなくクレジットから消費します。
無料トライアルキーは利用できません。

## インストール

### 1. API キー

[HelloCloud コンソール](https://router.hello.ai/console/) で有料 API キー (`hello_...`) を発行し、
環境変数 `HELLOCLOUD_API_KEY` に設定します。

```bash
export HELLOCLOUD_API_KEY=hello_xxxxxxxxxxxx
```

持続させる場合は、Hermes の `.env` (`~/.hermes/.env`、またはアクティブプロファイルの `.env`) に
`HELLOCLOUD_API_KEY=hello_...` を追記してください。

### 2. プラグイン

```bash
hermes plugins install elderica/hellocloud-hermes-agent-transcription --enable
```

後から有効化する場合:

```bash
hermes plugins enable hellocloud-transcription
```

### 3. プロバイダー設定

```bash
hermes config set stt.provider hellocloud
hermes config set stt.hellocloud.model whisper-large-v3-turbo --force
```

`config.yaml` 相当:

```yaml
stt:
  enabled: true
  provider: hellocloud
  hellocloud:
    model: whisper-large-v3-turbo
    # language: ja                 # 省略時は stt.language、未指定なら自動判定
    # base_url: https://api.router.hello.ai/v1
    # timeout: 300                 # 1 リクエストの秒数
    # max_retries: 1
    # temperature: 0               # 0〜1
```

設定変更は新規セッションで使用されます。CLI は再起動、ゲートウェイ / デスクトップは `/reset`
で新しいセッションを開始してください。

## 使い方

プラグインを選択すると、Hermes が受信した音声メッセージとボイスモードの録音を HelloCloud で文字起こしします。

```text
「この音声を文字起こしして」
```

言語は `stt.hellocloud.language` → `stt.language` の順で解決されます。日本語と分かっている場合は
`ja` を指定すると安定します。`stt.prompt` や `pre_transcription` フックのプロンプトは、
固有名詞のヒントとして API に転送されます。

## 制限とプライバシー

- 対応形式: mp3 / mp4 / m4a / wav / flac / ogg / webm
- 1 リクエストの上限は **25 MB** です。超過時は Hermes 本体が m4a へ再エンコードし、
  それでも超える場合は無音付近で分割して結果を結合します。ffmpeg が必要です。
- `response_format` は `verbose_json` を使用し、テキスト・再生時間・言語・`usage.seconds` を読みます。
  SRT / VTT の生成と単語単位のタイムスタンプ出力には対応していません。
- ストリーミング文字起こしには対応していません。`stt.streaming: false` で使用してください。
- 音声と文字起こし結果は HelloCloud のゼロデータ保持 (ZDR) 対象で、処理後に保存されません。
  利用履歴には課金秒数と料金だけが残ります。音声は API 処理のため HelloCloud に送信されます。

## 設定項目

| 場所 | キー | 既定 |
|------|------|------|
| 環境変数 | `HELLOCLOUD_API_KEY` | (必須) |
| `config.yaml` | `stt.provider` | `hellocloud` を指定 |
| `config.yaml` | `stt.hellocloud.model` | `whisper-large-v3-turbo` |
| `config.yaml` | `stt.hellocloud.language` | 未指定 (`stt.language` を継承) |
| `config.yaml` | `stt.hellocloud.base_url` | `https://api.router.hello.ai/v1` |
| `config.yaml` | `stt.hellocloud.timeout` | `300` 秒 |
| `config.yaml` | `stt.hellocloud.max_retries` | `1` |
| `config.yaml` | `stt.hellocloud.temperature` | 未指定 |

## テスト

Hermes のソースチェックアウトを `PYTHONPATH` に指定して実行します。ネットワークと実課金を伴わず、
OpenAI SDK はモックしています。

```bash
PYTHONPATH=/path/to/hermes-agent python -m unittest discover -s tests -v
python -m py_compile __init__.py
hermes plugins doctor --ci .
```

## ライセンス

[MIT](./LICENSE)
