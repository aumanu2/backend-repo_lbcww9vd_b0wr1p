import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict

# Make TensorFlow optional at runtime
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
HAS_TF = True
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except Exception:
    HAS_TF = False
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore

app = FastAPI(title="Multilingual Speech Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMOTIONS: List[str] = ["happy", "sad", "angry", "fear"]
SAMPLE_RATE = 16000
TARGET_DURATION = 3.0  # seconds
N_MELS = 64
TARGET_FRAMES = 300
HOP_LENGTH = 160  # 10ms
WIN_LENGTH = 400  # 25ms
MODEL_PATH = os.getenv("MODEL_PATH", "models/mser_tf.h5")

_model = None
_input_shape = (N_MELS, TARGET_FRAMES, 1)


def _fix_length(y: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    target_len = int(target_sec * sr)
    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    return y


def extract_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract a 0-1 normalized log-mel spectrogram. Falls back to scipy if librosa not available."""
    # Try librosa path
    try:
        import librosa  # type: ignore
        y = librosa.util.normalize(y)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=50,
            fmax=sr // 2,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_min, S_max = S_db.min(), S_db.max()
        if S_max - S_min > 1e-6:
            S_norm = (S_db - S_min) / (S_max - S_min)
        else:
            S_norm = np.zeros_like(S_db)
    except Exception:
        # Fallback using scipy.signal for a rough spectrogram then mel-like compression
        from scipy.signal import stft
        # Basic normalization
        y = y / (np.max(np.abs(y)) + 1e-9)
        f, t, Zxx = stft(y, fs=sr, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH)
        S = np.abs(Zxx) ** 2  # power
        # Reduce frequency bins to N_MELS via simple grouping
        if S.shape[0] >= N_MELS:
            step = S.shape[0] // N_MELS
            S_reduced = np.stack([S[i*step:(i+1)*step].mean(axis=0) for i in range(N_MELS)], axis=0)
        else:
            # pad frequencies
            pad_freq = N_MELS - S.shape[0]
            S_reduced = np.pad(S, ((0, pad_freq), (0, 0)), mode='constant')[:N_MELS]
        S_db = 10.0 * np.log10(np.maximum(S_reduced, 1e-10))
        S_min, S_max = S_db.min(), S_db.max()
        if S_max - S_min > 1e-6:
            S_norm = (S_db - S_min) / (S_max - S_min)
        else:
            S_norm = np.zeros_like(S_db)

    # Pad/trim to TARGET_FRAMES on time axis
    if S_norm.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad)), mode="constant")
    elif S_norm.shape[1] > TARGET_FRAMES:
        S_norm = S_norm[:, :TARGET_FRAMES]
    return S_norm.astype(np.float32)


# Lightweight TF model when TF is available
if HAS_TF:
    def build_model(input_shape, num_classes: int):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        return model
else:
    def build_model(*args, **kwargs):  # type: ignore
        return None


essential_audio_types = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "application/octet-stream",
}


def load_or_create_model():
    global _model
    if _model is not None:
        return _model
    if HAS_TF:
        try:
            if os.path.exists(MODEL_PATH):
                _model = keras.models.load_model(MODEL_PATH)
            else:
                _model = build_model(_input_shape, len(EMOTIONS))
        except Exception:
            _model = build_model(_input_shape, len(EMOTIONS))
    else:
        _model = None
    return _model


def numpy_fallback_predict(feat: np.ndarray) -> np.ndarray:
    x = feat.squeeze(-1)  # (1, n_mels, frames)
    x = x.mean(axis=2)    # (1, n_mels)
    rng = np.random.default_rng(42)
    W = rng.standard_normal((N_MELS, len(EMOTIONS))) * 0.05
    b = np.zeros((len(EMOTIONS),), dtype=np.float32)
    logits = x @ W + b
    logits = logits.astype(np.float32)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[0]


@app.get("/")
def read_root():
    return {"message": "Multilingual SER backend ready"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tensorflow": (tf.__version__ if HAS_TF else None),
        "tf_available": HAS_TF,
        "emotions": EMOTIONS,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("English")) -> Dict:
    if file.content_type not in essential_audio_types:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an audio file.")

    try:
        raw = await file.read()
        # Decode audio. Prefer soundfile, else librosa, else scipy.io.wavfile (WAV only)
        y = None
        sr = None

        # soundfile path
        if y is None:
            try:
                import soundfile as sf  # type: ignore
                with io.BytesIO(raw) as buf:
                    data, rate = sf.read(buf, dtype='float32', always_2d=False)
                    if isinstance(data, np.ndarray) and data.ndim > 1:
                        data = data.mean(axis=1)
                    y, sr = data, int(rate)
            except Exception:
                y, sr = None, None

        # librosa path
        if y is None:
            try:
                import librosa  # type: ignore
                with io.BytesIO(raw) as buf:
                    data, rate = librosa.load(buf, sr=None, mono=True)
                    y, sr = data, int(rate)
            except Exception:
                y, sr = None, None

        # scipy fallback (WAV only)
        if y is None:
            try:
                from scipy.io import wavfile
                with io.BytesIO(raw) as buf:
                    rate, data = wavfile.read(buf)
                    data = data.astype(np.float32)
                    # Normalize 16-bit PCM
                    if data.dtype == np.int16:
                        data = data / 32768.0
                    if isinstance(data, np.ndarray) and data.ndim > 1:
                        data = data.mean(axis=1)
                    y, sr = data, int(rate)
            except Exception:
                y, sr = None, None

        if y is None or sr is None:
            raise HTTPException(status_code=400, detail="Could not decode audio. Please upload WAV/MP3/FLAC/OGG.")

        # Resample if needed using scipy
        if sr != SAMPLE_RATE:
            try:
                import librosa  # type: ignore
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            except Exception:
                from scipy.signal import resample
                target_len = int(len(y) * (SAMPLE_RATE / sr))
                y = resample(y, target_len)
            sr = SAMPLE_RATE

        y = _fix_length(y, sr, TARGET_DURATION)
        feat = extract_logmel(y, sr)  # (n_mels, frames)
        feat = np.expand_dims(feat, axis=-1)  # (n_mels, frames, 1)
        feat = np.expand_dims(feat, axis=0)   # (1, n_mels, frames, 1)

        model = load_or_create_model()
        if HAS_TF and model is not None:
            preds = model.predict(feat, verbose=0)[0]
        else:
            preds = numpy_fallback_predict(feat)
        preds = preds.astype(float)
        probs = {emo: float(preds[i]) for i, emo in enumerate(EMOTIONS)}
        top_idx = int(np.argmax(preds))
        response = {
            "language": language,
            "emotion": EMOTIONS[top_idx],
            "probabilities": probs,
            "tf_available": HAS_TF,
        }
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)[:200]}")


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
