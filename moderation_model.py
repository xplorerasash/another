"""ModerationModel

This module exposes `ModerationModel.predict(text)` returning a stable dict
shape used by `moderation_engine.py`. It supports two backends:

- BERT (preferred): uses `transformers` to load a fine-tuned sequence
  classification model and returns a probability for the "harmful" class.
  The locally fine-tuned model in `models/bert_cyberbully` is used by
  default when it is present and complete; otherwise it falls back to the
  `unitary/toxic-bert` hub model.
- sklearn (fallback): preserves the original behavior loading a `joblib`
  pipeline produced by `train.py`.

The implementation is defensive: if `transformers` or `torch` are not
installed the class will transparently fall back to the sklearn pipeline.
"""
import logging
from pathlib import Path
from typing import Dict, Optional
import os

from utils.preprocess import clean_text

logger = logging.getLogger("safechat.moderation_model")
MAX_INPUT_LENGTH = 2000

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JOBLIB_PATH = BASE_DIR / "models" / "cyberbullying_model.joblib"
DEFAULT_HF_MODEL = "unitary/toxic-bert"
DEFAULT_LOCAL_MODEL = BASE_DIR / "models" / "bert_cyberbully"


def _is_complete_local_model(path: Path) -> bool:
    """A locally fine-tuned model is only usable if its weights, config,
    and tokenizer were all saved alongside each other."""
    return (
        (path / "config.json").exists()
        and (path / "model.safetensors").exists()
        and (path / "tokenizer.json").exists()
    )


def _resolve_model_id(model_id: Optional[str]) -> str:
    """Return an absolute model path for a relative path, else the input.

    With no explicit override, prefer the locally fine-tuned model
    (models/bert_cyberbully) when it exists and is complete; otherwise fall
    back to the well-established `unitary/toxic-bert` hub model.
    """
    if not model_id:
        if _is_complete_local_model(DEFAULT_LOCAL_MODEL):
            return str(DEFAULT_LOCAL_MODEL)
        return DEFAULT_HF_MODEL
    path = Path(model_id)
    if not path.is_absolute():
        candidate = BASE_DIR / path
        if candidate.exists():
            return str(candidate)
    return model_id


class ModerationModel:
    """Model abstraction with optional BERT backend.

    Args:
        model_path: If a string path to a joblib file is given it will load
            the sklearn pipeline. If a HuggingFace model id (str) is given and
            `transformers` is available, it will load that model/tokenizer.
        backend: Optional override of backend: 'bert' or 'sklearn'. If None
            the class will prefer BERT when available.
    """

    def __init__(self, model_path: Optional[Path] = None, backend: Optional[str] = None):
        self.joblib_path = Path(model_path) if model_path and str(model_path).endswith('.joblib') else DEFAULT_JOBLIB_PATH
        self.hf_model_id = None
        self._sklearn = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._backend = backend

        env_backend = os.getenv('MODERATION_MODEL_BACKEND')
        if env_backend:
            self._backend = env_backend

        if model_path and not str(model_path).endswith('.joblib'):
            self.hf_model_id = str(model_path)

        if self._backend is None and model_path and str(model_path).endswith('.joblib'):
            self._backend = 'sklearn'

        self._transformers_available = None

    # ---- Helpers ----
    def _has_transformers(self) -> bool:
        if self._transformers_available is not None:
            return self._transformers_available
        try:
            import transformers  # type: ignore
            import torch  # type: ignore
            self._transformers_available = True
        except Exception:
            self._transformers_available = False
        return self._transformers_available

    def _load_sklearn(self):
        if self._sklearn is None:
            import joblib
            if not self.joblib_path.exists():
                raise FileNotFoundError(f"No trained model found at {self.joblib_path}. Run `python train.py`.")
            self._sklearn = joblib.load(self.joblib_path)

    def _load_hf(self):
        if self._hf_model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_id = _resolve_model_id(self.hf_model_id or os.getenv('MODERATION_HF_MODEL', ''))
        self._hf_model_id = model_id
        logger.info("Loading HF model: %s", model_id)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._hf_model.eval()
        if torch.cuda.is_available():
            self._hf_model.to('cuda')
            logger.info("Model moved to CUDA")

    # ---- Public API ----
    def predict(self, text: str) -> Dict:
        if not isinstance(text, str) or not text.strip():
            return {"label": "safe", "is_harmful": False, "confidence": 0.0, "model_used": "none"}
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning("Input truncated from %d to %d chars", len(text), MAX_INPUT_LENGTH)
            text = text[:MAX_INPUT_LENGTH]
        cleaned = clean_text(text)
        if not cleaned.strip():
            return {"label": "safe", "is_harmful": False, "confidence": 0.0, "model_used": "none"}

        use_bert = False
        if self._backend == 'sklearn':
            use_bert = False
        elif self._backend == 'bert':
            use_bert = True
        else:
            use_bert = self._has_transformers()

        if use_bert:
            try:
                self._load_hf()
            except Exception:
                use_bert = False

        if use_bert:
            import torch

            inputs = self._hf_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                self._hf_model.to('cuda')
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                num_classes = len(probs)
                if num_classes == 2:
                    harmful_index = 1
                    confidence = float(probs[harmful_index])
                    predicted = int(probs.argmax())
                    label = 'harmful' if predicted == harmful_index else 'safe'
                else:
                    max_prob = float(probs.max())
                    confidence = max_prob
                    label = 'harmful' if max_prob >= 0.65 else 'safe'

                return {"label": label, "is_harmful": label == 'harmful', "confidence": confidence, "model_used": self._hf_model_id}

        self._load_sklearn()
        proba = self._sklearn.predict_proba([cleaned])[0]
        classes = list(self._sklearn.classes_)
        harmful_index = classes.index(1) if 1 in classes else (len(classes) - 1)
        confidence = float(proba[harmful_index])
        predicted_class = int(self._sklearn.predict([cleaned])[0])
        return {"label": "harmful" if predicted_class == 1 else "safe", "is_harmful": predicted_class == 1, "confidence": confidence, "model_used": str(self.joblib_path)}


_default_model = None


def get_model() -> ModerationModel:
    """Lazily-created module-level singleton.

    When transformers/torch are available and no explicit backend override
    is set, uses the locally fine-tuned BERT model (models/bert_cyberbully)
    when it is present and complete, otherwise the unitary/toxic-bert
    HuggingFace model for classification. Falls back to the sklearn
    pipeline otherwise.
    """
    global _default_model
    if _default_model is None:
        _default_model = ModerationModel()
    return _default_model
