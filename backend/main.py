"""
PaperTrap — FastAPI Backend (Fixed)
Run:  uvicorn main:app --reload
Deps: pip install fastapi uvicorn[standard] jinja2 python-multipart
             passlib[bcrypt] itsdangerous aiofiles
"""

from fastapi import FastAPI, Request, Form, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import bcrypt
import sqlite3
import os

# ── App setup ────────────────────────────────────────────────
app = FastAPI(title="PaperTrap")

# Session middleware 
app.add_middleware(
    SessionMiddleware,
    secret_key="papertrap-secret-key-change-in-production",
    session_cookie="papertrap_session",
    max_age=86400,        
    same_site="lax",
    https_only=False,      
)


os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



# Database
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            email    TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            created  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Auth helpers 
def get_current_user(request: Request) -> dict | None:
    """Return session user dict or None."""
    username = request.session.get("username")
    if not username:
        return None
    return {"username": username, "email": request.session.get("email", "")}

# Small helper for custom redirects
class HTTPException307(Exception):
    def __init__(self, location: str):
        self.location = location

@app.exception_handler(HTTPException307)
async def redirect_handler(request: Request, exc: HTTPException307):
    return RedirectResponse(exc.location, status_code=status.HTTP_303_SEE_OTHER)

# Template context helper 
def ctx(request: Request, **kwargs) -> dict:
    """Consolidates template context. Note: 'request' is required in the dict."""
    flash = request.session.pop("flash", None)
    return {
        "request": request,
        "user":    get_current_user(request),
        "flash":   flash,
        **kwargs,
    }

def flash_msg(request: Request, message: str, category: str = "info"):
    """Store a one-shot flash message in the session."""
    request.session["flash"] = {"msg": message, "category": category}

# Public routes 
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context=ctx(request)
    )

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        request=request, name="about.html", context=ctx(request)
    )

# Register 
@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request=request, name="register.html", context=ctx(request)
    )

@app.post("/register", response_class=HTMLResponse)
async def register_post(
    request:  Request,
    username: str = Form(...),
    email:    str = Form(...),
    password: str = Form(...),
    confirm:  str = Form(...),
):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)

    username = username.strip()
    email    = email.strip()
    errors   = []

    if not username or not email or not password:
        errors.append("All fields are required.")
    if password != confirm:
        errors.append("Passwords do not match.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")

    if errors:
        return templates.TemplateResponse(
            request=request, 
            name="register.html", 
            context=ctx(request, errors=errors, form={"username": username, "email": email})
        )

    try:
        conn = get_db()
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        conn.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (username, email, hashed_pw),
        )
        conn.commit()
        conn.close()
        flash_msg(request, f"Account created! Welcome, {username}. Please log in.", "success")
        return RedirectResponse("/login", status_code=303)
    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            request=request, 
            name="register.html", 
            context=ctx(request, errors=["Username or email already taken."], form={"username": username, "email": email})
        )

# Login 
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request=request, name="login.html", context=ctx(request)
    )

@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request:  Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)

    username = username.strip()
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        request.session["username"] = username
        request.session["email"]    = user["email"]
        flash_msg(request, f"Welcome back, {username}!", "success")
        next_url = request.session.pop("next", "/dashboard")
        return RedirectResponse(next_url, status_code=303)
    else:
        return templates.TemplateResponse(
            request=request, 
            name="login.html", 
            context=ctx(request, errors=["Invalid username or password."], form={"username": username})
        )

# Logout
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    flash_msg(request, "You have been logged out.", "success")
    return RedirectResponse("/login", status_code=303)

# Protected routes
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request=request, name="dashboard.html", context=ctx(request)
    )

@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    user = get_current_user(request)
    if not user:
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request=request, name="upload.html", context=ctx(request)
    )

@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    user = get_current_user(request)
    if not user:
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request=request, name="results.html", context=ctx(request)
    )

# Startup
@app.on_event("startup")
async def startup():
    init_db()