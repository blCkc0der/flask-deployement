import os
import base64
import tempfile
import json
from flask import Flask, request, jsonify, send_file, render_template
import requests
from google.cloud import texttospeech  # Google TTS

app = Flask(__name__)

# Load your OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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
                    {
                        "type": "text",
                        "text": "Analyse this photo and tell me what symptoms this person exhibits. Then tell me what diseases it might fall under."
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    return response

def text_to_speech_google(text):
    credentials_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    client = texttospeech.TextToSpeechClient.from_service_account_info(credentials_info)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        name="en-GB-Wavenet-D"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_audio.name, "wb") as out:
        out.write(response.audio_content)

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
        audio_path = text_to_speech_google(reply)

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
