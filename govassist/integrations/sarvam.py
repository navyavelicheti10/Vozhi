import logging
import mimetypes
import os
import base64
import io
import re
import wave
import requests
from typing import Any, Optional

from govassist.config import load_env_file

logger = logging.getLogger(__name__)

MIME_OVERRIDES = {
    ".m4a": "audio/x-m4a",
    ".webm": "audio/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

SUPPORTED_TTS_LANGUAGE_CODES = {
    "en-IN",
    "hi-IN",
    "bn-IN",
    "ta-IN",
    "te-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "gu-IN",
    "pa-IN",
    "od-IN",
}

# Maps short/ISO-639-1 codes and BCP-47 variants that Sarvam API may return
# to the expected "xx-IN" format used internally.
_LANGUAGE_CODE_ALIASES: dict[str, str] = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN",
    "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN",
    "pa": "pa-IN", "od": "od-IN", "or": "od-IN",
    # ISO-639-2 codes
    "tam": "ta-IN", "tel": "te-IN", "kan": "kn-IN",
    "mal": "ml-IN", "mar": "mr-IN", "guj": "gu-IN",
    "pan": "pa-IN", "ori": "od-IN", "hin": "hi-IN",
    "ben": "bn-IN", "eng": "en-IN",
}

class SarvamAIClient:
    """Wrapper for Sarvam AI chat, speech, translation, and TTS APIs."""
    
    def __init__(self):
        load_env_file()
        self.api_key = os.getenv("SARVAM_API_KEY", "")
        self.base_url = "https://api.sarvam.ai"
        self.chat_model = os.getenv("SARVAM_CHAT_MODEL", "sarvam-m").strip() or "sarvam-m"
        
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not set. Sarvam integration will be mocked/fail.")

    def _auth_headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "api-subscription-key": self.api_key,
            "Content-Type": content_type,
        }

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: int = 120,
    ) -> requests.Response:
        return requests.post(
            f"{self.base_url}{endpoint}",
            headers=self._auth_headers(),
            json=payload,
            timeout=timeout,
        )

    def _refresh_api_key(self) -> str:
        load_env_file()
        self.api_key = os.getenv("SARVAM_API_KEY", "").strip()
        return self.api_key

    def normalize_language_code(self, language_code: str | None) -> str:
        code = (language_code or "").strip()
        if not code:
            return "en-IN"
        # Exact match
        if code in SUPPORTED_TTS_LANGUAGE_CODES:
            return code
        # Try full alias lookup first (e.g. "tel", "ta", "pa")
        alias = _LANGUAGE_CODE_ALIASES.get(code)
        if alias:
            logger.debug("Normalized language code '%s' → '%s' via alias", code, alias)
            return alias
        # Try the primary subtag prefix (e.g. "te-Telu" → "te" → "te-IN")
        prefix = code.split("-")[0].lower()
        alias = _LANGUAGE_CODE_ALIASES.get(prefix)
        if alias:
            logger.debug("Normalized language code '%s' → '%s' via prefix '%s'", code, alias, prefix)
            return alias
        logger.warning("Unknown language code '%s', falling back to en-IN", code)
        return "en-IN"

    def _split_tts_segments(self, text: str, max_chars: int = 500) -> list[str]:
        normalized_text = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized_text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", normalized_text)
        segments: list[str] = []
        current = ""

        def flush_current() -> None:
            nonlocal current
            value = current.strip()
            if value:
                segments.append(value)
            current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                flush_current()
                words = sentence.split()
                chunk = ""
                for word in words:
                    candidate = f"{chunk} {word}".strip()
                    if len(candidate) <= max_chars:
                        chunk = candidate
                    else:
                        if chunk:
                            segments.append(chunk)
                        chunk = word[:max_chars]
                if chunk:
                    segments.append(chunk)
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush_current()
                current = sentence

        flush_current()
        return segments

    def _merge_wav_chunks(self, wav_chunks: list[bytes]) -> bytes:
        if not wav_chunks:
            return b""
        if len(wav_chunks) == 1:
            return wav_chunks[0]

        output_buffer = io.BytesIO()
        output_wave = None
        params = None

        try:
            for chunk in wav_chunks:
                with wave.open(io.BytesIO(chunk), "rb") as input_wave:
                    if params is None:
                        params = input_wave.getparams()
                        output_wave = wave.open(output_buffer, "wb")
                        output_wave.setparams(params)
                    elif input_wave.getparams()[:4] != params[:4]:
                        logger.warning("Sarvam TTS chunk parameters mismatched. Falling back to the first chunk.")
                        return wav_chunks[0]

                    output_wave.writeframes(input_wave.readframes(input_wave.getnframes()))
        except wave.Error as exc:
            logger.warning("Failed to merge Sarvam WAV chunks cleanly: %s", exc)
            return wav_chunks[0]
        finally:
            if output_wave is not None:
                output_wave.close()

        return output_buffer.getvalue()

    def _chunk_batches(self, items: list[str], batch_size: int = 3) -> list[list[str]]:
        return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]

    def _build_silence_wav(self, duration_ms: int = 500, sample_rate: int = 24000) -> bytes:
        frame_count = max(int(sample_rate * (duration_ms / 1000)), 1)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)
        return buffer.getvalue()

    def speech_to_text(self, audio_file_path: str, language_code: str = "unknown") -> str:
        """Transcribes incoming voice note."""
        result = self.speech_to_text_with_metadata(audio_file_path=audio_file_path, language_code=language_code)
        return result.get("transcript", "")

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        reasoning_effort: str | None = None,
        wiki_grounding: bool = False,
    ) -> str:
        api_key = self._refresh_api_key()
        normalized_messages = [
            {
                "role": (message.get("role") or "user").strip(),
                "content": (message.get("content") or "").strip(),
            }
            for message in messages
            if (message.get("content") or "").strip()
        ]
        if not normalized_messages:
            return ""

        if not api_key:
            logger.info("Mock Sarvam chat completion path hit because SARVAM_API_KEY is missing.")
            return normalized_messages[-1]["content"]

        payload: dict[str, Any] = {
            "messages": normalized_messages,
            "model": (model or self.chat_model or "sarvam-m").strip() or "sarvam-m",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "wiki_grounding": wiki_grounding,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        try:
            response = self._post_json("/v1/chat/completions", payload=payload, timeout=180)
            if response.status_code != 200:
                logger.error("Sarvam chat completion failed: %s", response.text)
                raise RuntimeError(f"Sarvam chat completion failed with status {response.status_code}")

            body = response.json()
            choices = body.get("choices") or []
            message = choices[0].get("message") if choices else {}
            content = (message or {}).get("content") or ""
            return content.strip()
        except Exception as exc:
            logger.error("Sarvam chat completion exception: %s", exc)
            raise

    def speech_to_text_with_metadata(self, audio_file_path: str, language_code: str = "unknown") -> dict[str, str]:
        """Transcribes incoming voice note and returns transcript plus detected language."""
        api_key = self._refresh_api_key()
        if not api_key:
            return {
                "transcript": "Mock STT: Provide PM Kisan details.",
                "language_code": "en-IN",
            }
            
        url = f"{self.base_url}/speech-to-text"
        
        try:
            suffix = os.path.splitext(audio_file_path)[1].lower()
            content_type = MIME_OVERRIDES.get(suffix) or mimetypes.guess_type(audio_file_path)[0] or "application/octet-stream"
            with open(audio_file_path, "rb") as f:
                files = {"file": (os.path.basename(audio_file_path), f, content_type)}
                data = {
                    "model": "saaras:v3",
                    "mode": "translate",
                    "language_code": language_code,
                    "prompt": "",
                }
                headers = {"api-subscription-key": api_key}
                logger.info("Sending audio file to Sarvam STT with content type: %s", content_type)
                response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
                
            if response.status_code == 200:
                payload = response.json()
                transcript = (payload.get("transcript") or "").strip()
                detected_language = payload.get("language_code")
                if transcript:
                    logger.info(
                        "Sarvam STT succeeded%s",
                        f" (detected language: {detected_language})" if detected_language else "",
                    )
                return {
                    "transcript": transcript,
                    "language_code": self.normalize_language_code(detected_language),
                }
            else:
                logger.error("Sarvam STT failed: %s", response.text)
                return {"transcript": "", "language_code": "en-IN"}
        except Exception as e:
            logger.error("Sarvam STT Exception: %s", e)
            return {"transcript": "", "language_code": "en-IN"}

    def translate_text(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str = "auto",
        mode: str = "modern-colloquial",
    ) -> dict[str, str]:
        api_key = self._refresh_api_key()
        normalized_text = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized_text:
            return {
                "translated_text": "",
                "source_language_code": self.normalize_language_code(source_language_code),
            }
        if not api_key:
            return {
                "translated_text": normalized_text,
                "source_language_code": self.normalize_language_code(source_language_code),
            }

        url = f"{self.base_url}/translate"
        try:
            response = self._post_json(
                "/translate",
                payload={
                    "input": normalized_text[:2000],
                    "source_language_code": source_language_code,
                    "target_language_code": target_language_code,
                    "mode": mode,
                    "model": "mayura:v1",
                },
                timeout=120,
            )
            if response.status_code != 200:
                logger.error("Sarvam translation failed: %s", response.text)
                return {
                    "translated_text": normalized_text,
                    "source_language_code": self.normalize_language_code(source_language_code),
                }

            payload = response.json()
            return {
                "translated_text": (payload.get("translated_text") or normalized_text).strip(),
                "source_language_code": self.normalize_language_code(payload.get("source_language_code")),
            }
        except Exception as exc:
            logger.error("Sarvam translation exception: %s", exc)
            return {
                "translated_text": normalized_text,
                "source_language_code": self.normalize_language_code(source_language_code),
            }

    def text_to_speech_bytes(self, text: str, language_code: str = "en-IN", speaker: str = "shubh") -> bytes:
        """Generates WAV audio bytes from text."""
        api_key = self._refresh_api_key()
        if not api_key:
            logger.info("Mock TTS: returning a short silent WAV clip.")
            return self._build_silence_wav()
            
        segments = self._split_tts_segments(text=text, max_chars=500)
        if not segments:
            return b""
        
        try:
            merged_batches: list[bytes] = []
            segment_batches = self._chunk_batches(segments, batch_size=3)

            for batch_index, segment_batch in enumerate(segment_batches, start=1):
                payload = {
                    "inputs": segment_batch,
                    "target_language_code": language_code,
                    "speaker": speaker,
                    "pace": 1.0,
                    "speech_sample_rate": 24000,
                    "enable_preprocessing": True,
                    "model": "bulbul:v3"
                }
                response = self._post_json("/text-to-speech", payload=payload, timeout=120)
                if response.status_code != 200:
                    logger.error("Sarvam TTS failed: %s", response.text)
                    return b""

                response_payload = response.json()
                audio_chunks = response_payload.get("audios", [])
                if not audio_chunks:
                    logger.error("Sarvam TTS returned no audio chunks for batch %s.", batch_index)
                    return b""

                decoded_chunks = [base64.b64decode(audio_chunk) for audio_chunk in audio_chunks if audio_chunk]
                merged_batches.append(self._merge_wav_chunks(decoded_chunks))

            logger.info(
                "Sarvam TTS succeeded for %s segment(s) across %s batch(es).",
                len(segments),
                len(segment_batches),
            )
            return self._merge_wav_chunks(merged_batches)
        except Exception as e:
            logger.error("Sarvam TTS Exception: %s", e)
            return b""

    def text_to_speech(self, text: str, output_file_path: str, language_code: str = "en-IN", speaker: str = "shubh"):
        """Backward-compatible file writer around byte generation."""
        audio_bytes = self.text_to_speech_bytes(text=text, language_code=language_code, speaker=speaker)
        with open(output_file_path, "wb") as f:
            f.write(audio_bytes)
        return output_file_path

sarvam_client = SarvamAIClient()
