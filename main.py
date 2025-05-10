

from flask import Flask, request, jsonify, send_file, render_template_string
from gtts import gTTS
import cv2
import time
import threading
from threading import Thread, Event
from io import BytesIO
import logging
import os
from flask_cors import CORS
import requests
import numpy as np
from PIL import Image
import pytesseract
import subprocess
import atexit
import json
import base64
import sys
from datetime import datetime
from flask import Response
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import httpx
import re

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

def call_gpt4o_with_image_sync(base64_image):
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer the question that this image shows, make sure I get awarded full marks for this A-level exam question. Format it in a way that makes it readable and understandable by text-to-speech. If the question requires drawing a graph, describe exactly how the graph should be drawn as I only have audio input, none visual."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    timeout = httpx.Timeout(200.0)
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(OPENAI_API_URL, headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }, json=data)
            return response
        except httpx.RequestError as e:
            logger.error(f"GPT-4o request error: {e}")
            time.sleep(2)
    raise Exception("Failed to call GPT-4o API after multiple attempts")

def clean_text_for_tts(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'[_~]', '', text)
    return text.strip()

def text_to_speech_openai_sync(text, filename):
    data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    timeout = httpx.Timeout(200.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", OPENAI_TTS_URL, headers=headers, json=data) as response:
                if response.status_code != 200:
                    error_message = b"".join(response.iter_bytes()).decode("utf-8")
                    raise Exception(f"TTS Error: {response.status_code} {error_message}")

                with open(filename, "wb") as f:
                    for chunk in response.iter_bytes():
                        if chunk:
                            f.write(chunk)
        return filename
    except Exception as e:
        logger.error(f"TTS API error: {str(e)}")
        return None
# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
PORT = 5000
NGROK_AUTH_TOKEN = "2qqvgcNkpXN0gXV5r5MwO1SgSCn_3k8UAFBfMxoZgoXSVMBhh" 
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for livestream processing
stream_active = Event()
current_stream_url = ""
frame_counter = 0
stream_lock = threading.Lock()

# Load pre-trained ResNet18 model
def load_model():
    try:
        
        model = models.resnet18(pretrained=True)
        model.eval()  

        with open('imagenet_classes.txt', 'r') as f:
            categories = [s.strip() for s in f.readlines()]

        return model, categories
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
       
        return None, ["Unknown"]

# load the model
try:
   
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with open('imagenet_classes.txt', 'w') as f:
            f.write(requests.get(url).text)

    model, categories = load_model()
    model_loaded = model is not None
except Exception as e:
    logger.error(f"Initial model loading failed: {str(e)}")
    model_loaded = False
    model = None
    categories = ["Unknown"]

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image):
    """Classify image using ResNet18 model"""
    try:
        if not model_loaded:
            return "Model not loaded", 0.0

        # Convert OpenCV image (BGR) to PIL Image (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess and create input tensor
        img_tensor = preprocess(pil_image)
        img_tensor = img_tensor.unsqueeze(0)  

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)

        # Get predicted class
        _, predicted = torch.max(output, 1)
        idx = predicted.item()

        # Get probability
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence = probabilities[idx].item() * 100

        # Return class name and confidence
        return categories[idx], confidence
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return "Classification failed", 0.0

def overlay_label(image, label, confidence):
    """Add prediction label to image"""
    try:
      
        result = image.copy()

        text = f"{label} ({confidence:.1f}%)"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = 10
        text_y = 30

        
        cv2.rectangle(result, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

        # Add text
        cv2.putText(result, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result
    except Exception as e:
        logger.error(f"Overlay error: {str(e)}")
        return image

def extract_text_from_image(image):
    """Enhanced OCR using pytesseract with preprocessing."""
    try:
       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

       
        text = pytesseract.image_to_string(gray)
        return text.strip() if text else "No text detected"
    except Exception as e:
        logger.error(f"OCR Error: {str(e)}")
        return "OCR processing failed"

def text_to_speech(text, lang='en'):
    """Convert text to audio with auto-play capability."""
    try:
        audio = BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        return None

def download_image(url):
    """Download image from any URL with enhanced error handling."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return None
    except Exception as e:
        logger.error(f"Image Download Error: {str(e)}")
        return None

def capture_stream_frame(stream_url, timeout=10):
    """
    Robust frame capture with enhanced error handling and connection management
    Args:
        stream_url: URL of the video stream
        timeout: Maximum time to wait for a frame (seconds)
    Returns:
        frame or None if capture fails
    """
    logger.info(f"Attempting to capture from: {stream_url}")
    
    # Clean up URL 
    stream_url = stream_url.strip()
    
    for attempt in range(3):
        try:
          
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)  
            
            if not cap.isOpened():
                logger.warning(f"Failed to open stream: {stream_url}")
                cap.release()
                time.sleep(1)
                continue
                
            # Configure capture properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
            
            
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    logger.info(f"Successfully captured frame from {stream_url}")
                    cap.release()
                    return frame
                time.sleep(0.2)
            
            logger.warning(f"Timeout waiting for frame from {stream_url}")
            cap.release()
            
        except Exception as e:
            logger.warning(f"Capture attempt {attempt+1} failed: {str(e)}")
            
            try:
                if 'cap' in locals() and cap is not None:
                    cap.release()
            except:
                pass
            time.sleep(1)

    logger.error(f"All capture attempts failed for {stream_url}")
    return None

def play_audio(audio_path):
    """Auto-play audio on default system speaker."""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(audio_path)
        else:  # macOS/Linux
            subprocess.run(['afplay' if sys.platform == 'darwin' else 'aplay', audio_path])
    except Exception as e:
        logger.error(f"Audio Playback Error: {str(e)}")

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_image_pipeline(image):
    """Complete classification → OCR → TTS pipeline."""
    try:
        # Classify with ResNet18
        label, confidence = classify_image(image)

        #  Add label overlay
        labeled_image = overlay_label(image, label, confidence)

        #  OCR 
        extracted_text = extract_text_from_image(image)

        analysis = f"Classification: {label} with {confidence:.1f}% confidence.\n"
        if extracted_text and extracted_text != "No text detected":
            analysis += f"\nText detected in image: {extracted_text}"

        #  Audio Conversion
        audio = text_to_speech(analysis)

       
        image_b64 = encode_image_to_base64(labeled_image)

        return analysis, audio, image_b64, labeled_image
    except Exception as e:
        logger.error(f"Pipeline Error: {str(e)}")
        return f"Processing error: {str(e)}", None, None, image

def process_livestream(stream_url, interval):
    global frame_counter

    while stream_active.is_set():
        try:
            frame = capture_stream_frame(stream_url)
            if frame is None:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(1)
                continue

            frame_counter += 1
            logger.info(f"Processing frame {frame_counter}...")

            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # Call GPT-4o
            gpt_response = call_gpt4o_with_image_sync(base64_image)
            if gpt_response.status_code == 200:
                gpt_reply = gpt_response.json()["choices"][0]["message"]["content"]
                clean_reply = clean_text_for_tts(gpt_reply)
            else:
                logger.error(f"GPT-4o error: {gpt_response.status_code}, {gpt_response.text}")
                clean_reply = "Error processing frame with GPT-4o."

            # Generate audio
            audio_path = os.path.join(UPLOAD_FOLDER, f"stream_{frame_counter}.mp3")
            tts_result = text_to_speech_openai_sync(clean_reply, audio_path)

            # Save frame image
            frame_path = os.path.join(UPLOAD_FOLDER, f"frame_{frame_counter}.jpg")
            cv2.imwrite(frame_path, frame)

            # Auto-play audio if generated
            if tts_result:
                play_audio(audio_path)

            logger.info(f"Frame {frame_counter} processed and audio generated.")
            time.sleep(interval)

        except Exception as e:
            logger.error(f"Stream Processing Error: {str(e)}")
            time.sleep(1)

def cleanup():
    """Cleanup resources when app exits."""
    logger.info("Shutting down gracefully...")
    stream_active.clear()
    try:
        ngrok.kill()
    except:
        pass

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ResNet18 Image Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f8fa; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #3498db; }
        input, button, select { padding: 10px; margin: 10px 0; border-radius: 4px; border: 1px solid #ddd; }
        button { background: #3498db; color: white; border: none; cursor: pointer; transition: background 0.3s; }
        button:hover { background: #2980b9; }
        button:disabled { background: #95a5a6; }
        #result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; display: none; }
        .result-image { max-width: 100%; margin: 10px 0; border-radius: 4px; }
        audio { width: 100%; margin-top: 10px; }
        #preview-image { max-width: 100%; margin: 10px 0; border-radius: 4px; display: none; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .loading:after {
            content: " .";
            animation: dots 1s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            40% { color: black; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            60% { text-shadow: .25em 0 0 black, .5em 0 0 rgba(0,0,0,0); }
            80%, 100% { text-shadow: .25em 0 0 black, .5em 0 0 black; }
        }
        .progress-container {
            margin-top: 10px;
            width: 100%;
            background-color: #f3f3f3;
            border-radius: 4px;
            display: none;
        }
        .progress-bar {
            height: 10px;
            border-radius: 4px;
            background-color: #3498db;
            width: 0%;
            transition: width 0.3s;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: bold;
        }
        .status-active {
            background-color: #2ecc71;
            color: white;
        }
        .status-idle {
            background-color: #95a5a6;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ResNet18 Image Analyzer</h1>

        <div id="model-status">
            <p>ResNet18 Model Status: <span id="model-status-text">Loading...</span></p>
        </div>

        <div>
            <h3>Single Image Analysis</h3>
            <input type="file" id="image-file" accept="image/*">
            <p>OR</p>
            <input type="text" id="image-url" placeholder="Enter image URL">
            <button id="analyze-btn">Analyze Image</button>
            <div class="loading" id="loading-single">Processing</div>
            <img id="preview-image">
        </div>

        <div id="livestream-controls">
            <h3>Livestream Processing</h3>
            <input type="text" id="stream-url" placeholder="Enter livestream URL (RTSP, HTTP, etc.)">
            <div>
                <label for="interval">Analysis Interval:</label>
                <select id="interval">
                    <option value="1">1 second</option>
                    <option value="5">5 seconds</option>
                    <option value="10" selected>10 seconds</option>
                    <option value="30">30 seconds</option>
                </select>
            </div>
            <button id="start-stream-btn">Start Processing</button>
            <button id="stop-stream-btn" disabled>Stop Processing</button>
            <p>Status: <span id="stream-status" class="status status-idle">Idle</span></p>
            <div class="progress-container" id="stream-progress-container">
                <div class="progress-bar" id="stream-progress"></div>
            </div>
        </div>

        <div id="result">
            <h3>Analysis Result:</h3>
            <img id="result-image" class="result-image">
            <p id="analysis-text"></p>
            <audio id="audio-player" controls autoplay></audio>
        </div>
    </div>

    <script>
        // Check model status when page loads
        document.addEventListener('DOMContentLoaded', async function() {
            try {
                const response = await fetch('/api/model_status');
                const data = await response.json();
                document.getElementById('model-status-text').textContent =
                    data.loaded ? 'Ready' : 'Failed to load (using fallback)';
            } catch (error) {
                document.getElementById('model-status-text').textContent = 'Error checking status';
            }
        });

        // Preview uploaded image
        document.getElementById('image-file').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    const img = document.getElementById('preview-image');
                    img.src = event.target.result;
                    img.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        // Single image analysis
        document.getElementById('analyze-btn').addEventListener('click', async () => {
            const fileInput = document.getElementById('image-file');
            const urlInput = document.getElementById('image-url').value;
            const resultDiv = document.getElementById('result');
            const loading = document.getElementById('loading-single');

            const formData = new FormData();

            if (fileInput.files.length > 0) {
                formData.append('image', fileInput.files[0]);
            } else if (urlInput) {
                formData.append('image_url', urlInput);
            } else {
                alert('Please upload an image or provide a URL');
                return;
            }

            try {
                // Show loading indicator
                loading.style.display = 'block';
                resultDiv.style.display = 'none';

                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('analysis-text').textContent = data.analysis;

                    // Display the analyzed image with label overlay
                    const resultImage = document.getElementById('result-image');
                    resultImage.src = data.image_url;

                    // Set up audio player
                    const audioPlayer = document.getElementById('audio-player');
                    audioPlayer.src = data.audio_url;
                    audioPlayer.play();

                    resultDiv.style.display = 'block';
                } else {
                    throw new Error(await response.text());
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            } finally {
                loading.style.display = 'none';
            }
        });

        // Livestream processing
        const startBtn = document.getElementById('start-stream-btn');
        const stopBtn = document.getElementById('stop-stream-btn');
        const statusSpan = document.getElementById('stream-status');
        const progressContainer = document.getElementById('stream-progress-container');
        const progressBar = document.getElementById('stream-progress');
        let eventSource;
        let intervalId;

        startBtn.addEventListener('click', async () => {
            const streamUrl = document.getElementById('stream-url').value;
            const interval = document.getElementById('interval').value;

            if (!streamUrl) {
                alert('Please enter a stream URL');
                return;
            }

            try {
                const response = await fetch('/api/start_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stream_url: streamUrl, interval: parseInt(interval) })
                });

                if (response.ok) {
                    statusSpan.textContent = "Processing";
                    statusSpan.className = "status status-active";
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    progressContainer.style.display = 'block';

                    // Update progress bar based on interval
                    let progress = 0;
                    progressBar.style.width = '0%';

                    intervalId = setInterval(() => {
                        progress = (progress + 1) % (parseInt(interval) + 1);
                        const percentage = (progress / parseInt(interval)) * 100;
                        progressBar.style.width = `${percentage}%`;
                    }, 1000);

                    // Listen for updates
                    eventSource = new EventSource('/api/stream_updates');

                    eventSource.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.image_url && data.audio_url) {
                            document.getElementById('analysis-text').textContent = data.analysis;
                            document.getElementById('result-image').src = data.image_url;

                            const audioPlayer = document.getElementById('audio-player');
                            audioPlayer.src = data.audio_url;
                            audioPlayer.play();

                            document.getElementById('result').style.display = 'block';

                            // Reset progress bar
                            progress = 0;
                            progressBar.style.width = '0%';
                        }
                    };
                } else {
                    throw new Error(await response.text());
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });

        stopBtn.addEventListener('click', async () => {
            try {
                await fetch('/api/stop_stream', { method: 'POST' });
                statusSpan.textContent = "Idle";
                statusSpan.className = "status status-idle";
                startBtn.disabled = false;
                stopBtn.disabled = true;
                progressContainer.style.display = 'none';

                if (intervalId) {
                    clearInterval(intervalId);
                }

                if (eventSource) {
                    eventSource.close();
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/model_status')
def model_status():
    """Check if ResNet18 model is loaded"""
    return jsonify({"loaded": model_loaded})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Handle URL
        elif 'image_url' in request.form:
            url = request.form['image_url']
            image = download_image(url)
            if image is None:
                return jsonify({"error": "Could not download image from URL"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400

       
        analysis, audio, image_b64, labeled_image = process_image_pipeline(image)

        if not audio:
            return jsonify({"error": "Audio generation failed"}), 500

       
        timestamp = int(time.time())
        image_path = os.path.join(UPLOAD_FOLDER, f"result_{timestamp}.jpg")
        audio_path = os.path.join(UPLOAD_FOLDER, f"result_{timestamp}.mp3")

        cv2.imwrite(image_path, labeled_image)
        with open(audio_path, 'wb') as f:
            f.write(audio.getvalue())

        return jsonify({
            "analysis": analysis,
            "audio_url": f"/audio/result_{timestamp}.mp3",
            "image_url": f"/images/result_{timestamp}.jpg"
        })

    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    global stream_active, current_stream_url, frame_counter

    with stream_lock:
        try:
            data = request.json
            stream_url = data.get('stream_url')
            interval = int(data.get('interval', 10))

            if not stream_url:
                return jsonify({"error": "No stream URL provided"}), 400

            if stream_active.is_set():
                return jsonify({"error": "Stream processing already active"}), 400

            current_stream_url = stream_url
            stream_active.set()
            frame_counter = 0

            Thread(
                target=process_livestream,
                args=(stream_url, interval),
                daemon=True
            ).start()

            return jsonify({
                "status": "success",
                "message": f"Stream processing started",
                "interval": interval,
                "stream_url": stream_url
            })

        except Exception as e:
            logger.error(f"Stream Start Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    global stream_active
    stream_active.clear()
    return jsonify({"status": "success", "message": "Stream processing stopped"})

@app.route('/api/stream_updates')
def stream_updates():
    """Server-Sent Events (SSE) for live updates"""
    def event_stream():
        last_frame = 0
        try:
            while True:
                global frame_counter

                if not stream_active.is_set():
                    yield f"data: {json.dumps({'status': 'stopped'})}\n\n"
                    break

                
                if frame_counter > last_frame:
                    last_frame = frame_counter

                    # Create update data
                    update_data = {
                        "status": "active",
                        "frame_counter": frame_counter,
                        "image_url": f"/images/frame_{frame_counter}.jpg",
                        "audio_url": f"/audio/stream_{frame_counter}.mp3",
                        "analysis": f"Frame {frame_counter} analysis completed"
                    }

                    yield f"data: {json.dumps(update_data)}\n\n"

                time.sleep(0.5)
        except Exception as e:
            logger.error(f"SSE Error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={'Cache-Control': 'no-cache'}
    )

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        return send_file(
            os.path.join(UPLOAD_FOLDER, filename),
            mimetype="audio/mpeg",
            as_attachment=False
        )
    except FileNotFoundError:
        return "Audio not found", 404

@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_file(
            os.path.join(UPLOAD_FOLDER, filename),
            mimetype="image/jpeg",
            as_attachment=False
        )
    except FileNotFoundError:
        return "Image not found", 404

def start_ngrok():
    """Start ngrok tunnel and return public URL."""
    try:
        
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        ngrok.kill()

        http_tunnel = ngrok.connect(PORT)
        public_url = http_tunnel.public_url

       
        logger.info(f"Ngrok started: {public_url}")
        print(f"\n{'='*60}")
        print(f" Public URL: {public_url}")
        print(f"{'='*60}\n")

        return public_url
    except Exception as e:
        logger.error(f"Ngrok Error: {str(e)}")
        print(f" Failed to start ngrok: {str(e)}")
        return None

if __name__ == "__main__":
    
    atexit.register(cleanup)

    

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    
    app.run(host='0.0.0.0', port=PORT)
