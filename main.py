import os
import base64
import tempfile
from flask import Flask, request, jsonify, send_file, render_template
import requests

app = Flask(__name__)

OPENAI_API_KEY = "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
"
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
                    { "type": "text", "text": "Analyse this photo and tell me what symptoms this person exhibits. Then tell me what diseases it might fall under" },
                    { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" } }
                ]
            }
        ],
        "max_tokens": 1000
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    return response

def text_to_speech_openai(text):
    tts_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    tts_data = {
        "model": "tts-1",  # or "tts-1-hd" if available
        "input": text,
        "voice": "nova",   # or "onyx", "shimmer", "echo", etc.
        "response_format": "mp3"
    }

    response = requests.post(OPENAI_TTS_URL, headers=tts_headers, json=tts_data)

    if response.status_code != 200:
        raise Exception(f"TTS failed: {response.status_code}, {response.text}")

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(response.content)
    temp_audio.close()
    return temp_audio.name

@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    base64_image = base64.b64encode(file.read()).decode("utf-8")
    gpt_response = call_gpt4o_with_image(base64_image)

    if gpt_response.status_code == 200:
        reply = gpt_response.json()["choices"][0]["message"]["content"]
        audio_path = text_to_speech_openai(reply)

        return send_file(
            audio_path,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="response.mp3"
        )
    else:
        return jsonify({
            "error": "Failed to get response from GPT",
            "details": gpt_response.text
        }), 500

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
