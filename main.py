import os
import base64
import asyncio
import tempfile
from flask import Flask, request, jsonify, send_file, render_template # to render htmml template for your website
import requests
import edge_tts  # Microsoft TTS


app = Flask(__name__)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")   # Your real key. put it in render enironment variable.
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
                    { "type": "text", "text": "Analyse this photo and tell me what symptoms this person exhibits. Then tell me what diseases it might fall under" },
                    { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" } }
                ]
            }
        ],
        "max_tokens": 1000
    }


    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    return response


async def text_to_speech_edge(text):
    voice = "en-GB-SoniaNeural"  # British female voice, change if you want
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(temp_audio.name)
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


        # Run the async edge-tts in sync context
        audio_path = asyncio.run(text_to_speech_edge(reply))


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

