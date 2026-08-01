"""SafeChat-AI REST API built with FastAPI.

Endpoints:
  POST   /api/register          Register a new user
  POST   /api/login             Login, receive JWT token
  POST   /api/chat              Send a message (requires JWT)
  GET    /api/admin/violations  List all violations (admin key required)
  DELETE /api/admin/reset/{uid} Reset violations for a user (admin key required)
  GET    /                      Serve the HTML chat interface
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from chatbot_impl import process_message
from violation_manager import ViolationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("safechat")

# ---- Config ----
SECRET_KEY = "safechat-dev-secret-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
ADMIN_KEY = "admin-secret-key"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="SafeChat-AI", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

USERS_FILE = BASE_DIR / "models" / "users.json"
tracker = ViolationManager()

# ---- Pydantic models ----

class RegisterRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 4:
            raise ValueError("Password must be at least 4 characters")
        return v

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 2000:
            raise ValueError("Message too long (max 2000 characters)")
        return v

# ---- User helpers ----

def _load_users() -> Dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except (json.JSONDecodeError, Exception):
            return {}
    return {}

def _save_users(users: Dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))

def _create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---- Endpoints ----

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = BASE_DIR / "templates" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>SafeChat-AI</h1><p>UI template not found.</p>", status_code=200)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.post("/api/register")
def register(body: RegisterRequest):
    users = _load_users()
    username = body.username.strip().lower()
    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    users[username] = {
        "password": pwd_context.hash(body.password),
        "created_at": time.time(),
    }
    _save_users(users)
    token = _create_token({"sub": username})
    logger.info("User registered: %s", username)
    return {"access_token": token, "token_type": "bearer", "username": username}

@app.post("/api/login")
def login(body: LoginRequest):
    users = _load_users()
    username = body.username.strip().lower()
    user = users.get(username)
    if not user or not pwd_context.verify(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _create_token({"sub": username})
    logger.info("User logged in: %s", username)
    return {"access_token": token, "token_type": "bearer", "username": username}

@app.post("/api/chat")
@limiter.limit("30/minute")
def chat(request: Request, body: ChatRequest, username: str = Depends(_verify_token)):
    start = time.time()
    result = process_message(username, body.message)
    elapsed = time.time() - start
    logger.info("Chat | user=%s len=%d type=%s elapsed=%.2fs", username, len(body.message), result["type"], elapsed)
    return result

@app.get("/api/admin/violations")
def admin_violations(key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return {"violations": tracker._data}

@app.delete("/api/admin/reset/{user_id}")
def admin_reset(user_id: str, key: str = ""):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    tracker.reset_user(user_id)
    logger.info("Admin reset violations for user: %s", user_id)
    return {"status": "ok", "user": user_id}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "SafeChat-AI", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
