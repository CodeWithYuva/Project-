from fastapi import FastAPI
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io

app = FastAPI()

class AudioRequest(BaseModel):
    audio_base64: str
    language: str

@app.post("/detect")
def detect_voice(data: AudioRequest):
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc)

        # 🔴 Simple AI vs Human logic (prototype)
        if mfcc_mean < -200:
            label = "AI-GENERATED"
            confidence = 0.87
        else:
            label = "HUMAN"
            confidence = 0.82

        return {
            "status": "success",
            "language": data.language,
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }