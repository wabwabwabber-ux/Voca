import json
import logging
import re
import os
import socket
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError

warnings.simplefilter("ignore", FutureWarning)

import google.generativeai as genai
from deep_translator import GoogleTranslator
from flask import Flask, Response, jsonify, render_template, request, stream_with_context


API_KEY = os.environ.get("GEMINI_API_KEY")

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voca")

PRIMARY_MODEL_NAME = "gemini-3-flash"
FALLBACK_MODEL_NAMES = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

REQUEST_TIMEOUT_SECONDS = 45
TRANSLATION_TIMEOUT_SECONDS = 18
executor = ThreadPoolExecutor(max_workers=10)

LANGUAGES = {
    "amharic": {"label": "Amharic", "translator": "am", "flag": "🇪🇹"},
    "oromo": {"label": "Oromo", "translator": "om", "flag": "🇪🇹"},
    "somali": {"label": "Somali", "translator": "so", "flag": "🇸🇴"},
    "swahili": {"label": "Swahili", "translator": "sw", "flag": "🇰🇪"},
    "bengali": {"label": "Bengali", "translator": "bn", "flag": "🇧🇩"},
    "urdu": {"label": "Urdu", "translator": "ur", "flag": "🇵🇰"},
    "tamil": {"label": "Tamil", "translator": "ta", "flag": "🇮🇳"},
    "burmese": {"label": "Burmese", "translator": "my", "flag": "🇲🇲"},
    "khmer": {"label": "Khmer", "translator": "km", "flag": "🇰🇭"},
    "kazakh": {"label": "Kazakh", "translator": "kk", "flag": "🇰🇿"},
}

SENTENCE_PATTERN = re.compile(r"(.+?[.!?\n])(\s+|$)", re.DOTALL)


class VocaError(Exception):
    def __init__(self, message, stage="Application", status_code=500, detail=None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.status_code = status_code
        self.detail = detail


def clean_error_detail(error):
    detail = str(error) or error.__class__.__name__
    if API_KEY:
        detail = detail.replace(API_KEY, "[hidden]")
    return detail[:900]


def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def run_with_timeout(label, fn, timeout, stage):
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except TimeoutError as exc:
        future.cancel()
        raise VocaError(
            f"{label} took too long.",
            stage=stage,
            status_code=504,
            detail=f"Timed out after {timeout} seconds.",
        ) from exc
    except VocaError:
        raise
    except Exception as exc:
        logger.warning("%s failed: %s", label, clean_error_detail(exc))
        raise VocaError(
            f"{label} failed.",
            stage=stage,
            status_code=502,
            detail=clean_error_detail(exc),
        ) from exc


def language_config(language_key):
    config = LANGUAGES.get((language_key or "").lower())
    if not config:
        raise VocaError("Unsupported language selected.", stage="Language", status_code=400)
    return config


def translate_text(text, source, target):
    if not text.strip():
        return ""

    def translate():
        return GoogleTranslator(source=source, target=target).translate(text)

    translated = run_with_timeout(
        "Translation",
        translate,
        TRANSLATION_TIMEOUT_SECONDS,
        stage=f"Translation: {source} to {target}",
    )

    if not translated:
        raise VocaError(
            "Translation returned no text.",
            stage=f"Translation: {source} to {target}",
            status_code=502,
            detail="The free GoogleTranslator endpoint returned an empty response.",
        )

    return translated


def configure_gemini():
    if not API_KEY or API_KEY == "YOUR_KEY_HERE":
        raise VocaError(
            "Gemini API key is missing.",
            stage="Gemini",
            status_code=500,
            detail="Set API_KEY at the top of app.py.",
        )
    genai.configure(api_key=API_KEY)


def make_model(model_name):
    configure_gemini()
    return genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": 0.55,
            "top_p": 0.9,
            "max_output_tokens": 900,
        },
    )


def gemini_token_stream(english_prompt):
    model_names = [PRIMARY_MODEL_NAME, *FALLBACK_MODEL_NAMES]
    prompt = (
        "You are Voca, a bridge for global communication. "
        "Answer clearly, naturally, and concisely in English. "
        "The answer will be translated sentence by sentence into the user's chosen language, "
        "so prefer complete sentences and avoid markdown tables.\n\n"
        f"User message:\n{english_prompt}"
    )

    errors = []
    for model_name in model_names:
        try:
            model = make_model(model_name)
            stream = model.generate_content(
                prompt,
                stream=True,
                request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
            )

            yielded_anything = False
            for chunk in stream:
                text = extract_chunk_text(chunk)
                if text:
                    yielded_anything = True
                    yield model_name, text

            if yielded_anything:
                return

            errors.append(f"{model_name}: empty stream")
        except Exception as exc:
            detail = clean_error_detail(exc)
            errors.append(f"{model_name}: {detail}")
            can_fallback = (
                model_name != model_names[-1]
                and (
                    "not found" in detail.lower()
                    or "not supported" in detail.lower()
                    or "404" in detail
                )
            )
            if can_fallback:
                continue
            raise VocaError(
                "Gemini streaming failed.",
                stage=f"Gemini: {model_name}",
                status_code=502,
                detail=detail,
            ) from exc

    raise VocaError(
        "No Gemini Flash model returned a stream.",
        stage="Gemini",
        status_code=502,
        detail=" | ".join(errors),
    )


def extract_chunk_text(chunk):
    try:
        if getattr(chunk, "text", None):
            return chunk.text
    except Exception:
        pass

    pieces = []
    for candidate in getattr(chunk, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", "")
            if text:
                pieces.append(text)
    return "".join(pieces)


def pop_complete_sentence(buffer):
    match = SENTENCE_PATTERN.search(buffer)
    if not match:
        return None, buffer
    sentence = match.group(1).strip()
    remaining = buffer[match.end() :].lstrip()
    return sentence, remaining


def stream_voca_response(message, language_key):
    try:
        config = language_config(language_key)
        language_label = config["label"]
        language_code = config["translator"]

        yield sse("status", {"message": f"Listening in {language_label}. Translating to English..."})
        english_input = translate_text(message, source=language_code, target="en")
        yield sse("source", {"english": english_input})

        yield sse("status", {"message": "Voca is generating a streamed response..."})

        buffer = ""
        model_used = None
        for model_name, token in gemini_token_stream(english_input):
            if model_used is None:
                model_used = model_name
                yield sse("model", {"model": model_used})

            buffer += token
            while True:
                sentence, buffer = pop_complete_sentence(buffer)
                if not sentence:
                    break

                translated = translate_text(sentence, source="en", target=language_code)
                yield sse(
                    "chunk",
                    {
                        "text": translated,
                        "english": sentence,
                        "model": model_used,
                    },
                )

        if buffer.strip():
            sentence = buffer.strip()
            translated = translate_text(sentence, source="en", target=language_code)
            yield sse(
                "chunk",
                {
                    "text": translated,
                    "english": sentence,
                    "model": model_used,
                },
            )

        yield sse("done", {"message": "Complete", "model": model_used})
    except VocaError as exc:
        yield sse(
            "voca-error",
            {
                "message": exc.message,
                "stage": exc.stage,
                "detail": exc.detail,
            },
        )
    except Exception as exc:
        logger.exception("Unexpected streaming failure")
        yield sse(
            "voca-error",
            {
                "message": "Something unexpected happened.",
                "stage": "Application",
                "detail": clean_error_detail(exc),
            },
        )


@app.get("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "primary_model": PRIMARY_MODEL_NAME,
            "fallback_models": FALLBACK_MODEL_NAMES,
            "languages": list(LANGUAGES.keys()),
        }
    )


@app.get("/api/stream")
def stream_route():
    message = (request.args.get("message") or "").strip()
    language = (request.args.get("language") or "amharic").strip()

    if not message:
        return jsonify({"error": "Please speak or type something first."}), 400

    if len(message) > 4000:
        return jsonify({"error": "Please keep input under 4,000 characters."}), 413

    response = Response(
        stream_with_context(stream_voca_response(message, language)),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route not found."}), 404


def pick_port(preferred_port=5000):
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=pick_port(), debug=True, threaded=True)
