import os
import base64
import time
import re
from flask import Flask, request, jsonify, send_file, render_template
import httpx

app = Flask(__name__)

OPENAI_API_KEY = "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"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

def clean_text_for_tts(text):
    text = re.sub(r'\*+', '', text)  # remove markdown bold/italic
    text = re.sub(r'[_`~]', '', text)  # remove code symbols
    return text.strip()

def call_gpt4o_with_image_sync(base64_image):
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

    timeout = httpx.Timeout(200.0)
    retries = 3
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(OPENAI_API_URL, headers=headers, json=data)
            return response
        except httpx.RequestError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    raise Exception("Failed to call GPT API after multiple attempts")

def text_to_speech_openai_sync(text):
    print(f"TTS input text: '{text}'")

    tts_data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }
    tts_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    output_path = "/tmp/response.mp3"
    timeout = httpx.Timeout(200.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENAI_TTS_URL, headers=tts_headers, json=tts_data, stream=True)

        print(f"TTS API Response Status: {response.status_code}")
        print(f"TTS API Response Headers: {response.headers}")
        print(f"TTS API Response Content (first 100 bytes): {response.content[:100]}")

        if response.status_code != 200:
            raise Exception(f"TTS Error: {response.status_code} {response.text}")

        # Write the audio content
        with open(output_path, "wb") as f:
            f.write(response.content)

        if os.path.getsize(output_path) == 0:
            print("TTS API returned an empty audio file.")
            raise Exception("TTS API returned an empty audio file")

        return output_path

    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        raise

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        print("Received a request to /upload")

        file = request.files.get("image")
        if not file:
            print("No image uploaded")
            return jsonify({"error": "No image uploaded"}), 400

        print("Reading and encoding image...")
        base64_image = base64.b64encode(file.read()).decode("utf-8")
        print("Image successfully converted to Base64")

        print("Calling GPT API...")
        gpt_response = call_gpt4o_with_image_sync(base64_image)
        print(f"GPT API response status: {gpt_response.status_code}")

        if gpt_response.status_code == 200:
            reply = gpt_response.json()["choices"][0]["message"]["content"]
            print("Raw GPT reply:", reply)

            clean_reply = clean_text_for_tts(reply)
            print("Cleaned reply for TTS:", clean_reply)

            print("Calling TTS API...")
            audio_path = text_to_speech_openai_sync(clean_reply)
            print(f"Audio file generated at: {audio_path}")

            return send_file(
                audio_path,
                mimetype="audio/mpeg",
                as_attachment=False,
                download_name="response.mp3"
            )
        else:
            print(f"GPT API Error: {gpt_response.status_code}, {gpt_response.text}")
            return jsonify({
                "error": "Failed to get response from GPT",
                "details": gpt_response.text
            }), 500
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route("/test_openai", methods=["GET"])
def test_openai():
    try:
        response = httpx.get("https://api.openai.com/v1/models", headers=headers)
        print(f"Test OpenAI API Response: {response.status_code}")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
