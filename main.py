import io
import base64
import asyncio
import tempfile
import pyttsx3  # Offline TTS library
from flask import Flask, request, jsonify, send_file
import requests

app = Flask(__name__)

OPENAI_API_KEY = "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"  # Put your real OpenAI API key here
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
                    {"type": "text", "text": "Answer the question that this image shows, make sure I get awarded full marks for this A-level exam question. Format it in a way that makes it readable and understandable by text-to-speech."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    return response

def text_to_speech_pyttsx3(text):
    engine = pyttsx3.init()  # Initialize the pyttsx3 engine
    engine.save_to_file(text, 'response.mp3')  # Save audio to file
    engine.runAndWait()  # Wait for the speech to finish
    return 'response.mp3'  # Return the file path

@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    base64_image = base64.b64encode(file.read()).decode("utf-8")
    gpt_response = call_gpt4o_with_image(base64_image)

    if gpt_response.status_code == 200:
        reply = gpt_response.json()["choices"][0]["message"]["content"]

        # Run the offline TTS (pyttsx3)
        audio_path = text_to_speech_pyttsx3(reply)

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

if __name__ == "__main__":
    app.run(debug=True)
