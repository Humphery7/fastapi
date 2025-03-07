from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import torch
from transformers import pipeline
import librosa
from huggingface_hub import login

#get token
TOKEN = os.getenv("TOKEN")
login(token=TOKEN)


# Initialize FastAPI app
app = FastAPI()

# Set up device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ASR model
pipe = pipeline(
    task='automatic-speech-recognition',
    model='Humphery7/yoruba-dataaugmentation',
    chunk_length_s=25,
    device=device
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="yoruba", task="transcribe")

# Route to return the index.html file
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("templates/index.html")
    return html_path.read_text(encoding="utf-8")

# Transcription function
async def transcribe(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    prediction = pipe(audio, batch_size=8)["text"]
    return prediction

# Audio transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), res: Response = Response()):
    try:
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
            transcription = await transcribe(temp_file_path)

        # Remove the temporary file after transcription
        os.remove(temp_file_path)

        return {"transcription": transcription}

    except Exception as error:
        print(f"Error during transcription: {str(error)}")
        res.status_code = 500
        return {"error": "Internal Server Error"}
