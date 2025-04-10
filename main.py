import os
import io
import base64
import asyncio
import tempfile
# import pyttsx3  # Offline TTS library removed for compatibility
from flask import Flask, request, jsonify, send_file, render_template
import requests

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}


def call_gpt4o_with_image(base64_image):
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer the question that this image shows, make sure I get awarded full marks for this A-level exam question. Format it in a way that makes it readable and understandable by text-to-speech."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    return response


def text_to_speech_openai(text):
    tts_data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }
    tts_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(OPENAI_TTS_URL, headers=tts_headers, json=tts_data, stream=True)
    if response.status_code == 200:
        output_path = "/tmp/response.mp3"
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_path
    else:
        print(f"TTS response content: {response.text}")  # Log response
        raise Exception(f"TTS Error: {response.status_code} {response.text}")


@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    base64_image = base64.b64encode(file.read()).decode("utf-8")
    gpt_response = call_gpt4o_with_image(base64_image)

    if gpt_response.status_code == 200:
        reply = gpt_response.json()["choices"][0]["message"]["content"]

        try:
            # Run OpenAI TTS
            audio_path = text_to_speech_openai(reply)

            return send_file(
                audio_path,
                mimetype="audio/mpeg",
                as_attachment=False,
                download_name="response.mp3"
            )
        except Exception as e:
            return jsonify({"error": "TTS conversion failed", "details": str(e)}), 500

    else:
        return jsonify({
            "error": "Failed to get response from GPT",
            "details": gpt_response.text
        }), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
