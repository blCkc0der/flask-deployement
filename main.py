import os
import io
import base64
import asyncio
import tempfile
import time
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


async def call_gpt4o_with_image(base64_image):
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

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=data)
    return response


async def text_to_speech_openai(text):
    tts_data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }
    tts_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_TTS_URL, headers=tts_headers, json=tts_data, stream=True)

    if response.status_code == 200:
        output_path = "/tmp/response.mp3"
        with open(output_path, "wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)
        return output_path
    else:
        raise Exception(f"TTS Error: {response.status_code} {response.text}")


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

    timeout = httpx.Timeout(60.0)  # 60 seconds
    retries = 3
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(OPENAI_API_URL, headers=headers, json=data)
            return response
        except httpx.RequestError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    raise Exception("Failed to call GPT API after multiple attempts")


def text_to_speech_openai_sync(text):
    tts_data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }
    tts_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    timeout = httpx.Timeout(60.0)  # 60 seconds
    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENAI_TTS_URL, headers=tts_headers, json=tts_data)

    if response.status_code == 200:
        output_path = "/tmp/response.mp3"
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        return output_path
    else:
        raise Exception(f"TTS Error: {response.status_code} {response.text}")


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
            print("Extracting GPT response content...")
            reply = gpt_response.json()["choices"][0]["message"]["content"]

            print("Calling TTS API...")
            audio_path = text_to_speech_openai_sync(reply)
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
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
