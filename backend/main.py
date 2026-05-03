"""
PaperTrap — FastAPI Backend
Run:  uvicorn main:app --reload
Deps: pip install fastapi uvicorn[standard] jinja2 python-multipart passlib[bcrypt] itsdangerous
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import bcrypt
import sqlite3, json, dataclasses

from ml.shap_explainer import compute_shap_explanation, ShapResult
from ml.adversarial    import run_adversarial_tests, AdversarialReport

# ── Lifespan (replaces deprecated @app.on_event) ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(title="PaperTrap", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key="papertrap-secret-key-change-in-production",
    session_cookie="papertrap_session",
    max_age=86400,
    same_site="lax",
    https_only=False,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Database ─────────────────────────────────────────────────
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

# ── Auth helpers ─────────────────────────────────────────────
def get_current_user(request: Request) -> dict | None:
    username = request.session.get("username")
    if not username:
        return None
    return {"username": username, "email": request.session.get("email", "")}

def flash_msg(request: Request, message: str, category: str = "info"):
    request.session["flash"] = {"msg": message, "category": category}

# ── Template render helper ────────────────────────────────────
# Starlette >= 0.28 NEW API:
#   templates.TemplateResponse(request, "page.html", context={...})
# OLD broken API:
#   templates.TemplateResponse("page.html", {"request": req, ...})  ← TypeError
def render(request: Request, template: str, **kwargs) -> HTMLResponse:
    flash = request.session.pop("flash", None)
    context = {
        "request": request,
        "user":    get_current_user(request),
        "flash":   flash,
        **kwargs,
    }
    return templates.TemplateResponse(request, template, context)

# ── Public routes ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render(request, "index.html")

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return render(request, "about.html")

# ── Register ──────────────────────────────────────────────────
@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return render(request, "register.html")

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
    errors: list[str] = []

    if not username or not email or not password:
        errors.append("All fields are required.")
    if password != confirm:
        errors.append("Passwords do not match.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")

    if errors:
        return render(request, "register.html",
                      errors=errors,
                      form={"username": username, "email": email})
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
        return render(request, "register.html",
                      errors=["Username or email already taken."],
                      form={"username": username, "email": email})

# ── Login ─────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return render(request, "login.html")

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
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode('utf-8'), row["password"].encode('utf-8')):
        request.session["username"] = username
        request.session["email"]    = row["email"]
        flash_msg(request, f"Welcome back, {username}!", "success")
        next_url = request.session.pop("next", "/dashboard")
        return RedirectResponse(next_url, status_code=303)

    return render(request, "login.html",
                  errors=["Invalid username or password."],
                  form={"username": username})

# ── Logout ────────────────────────────────────────────────────
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    flash_msg(request, "You have been logged out.", "success")
    return RedirectResponse("/login", status_code=303)

# ── Protected routes ──────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not get_current_user(request):
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)
    return render(request, "dashboard.html")

@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    if not get_current_user(request):
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)
    return render(request, "upload.html")

@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    if not get_current_user(request):
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)
    return render(request, "results.html")


# ── SHAP Explainability Dashboard ─────────────────────────────

def _dataclass_to_dict(obj) -> dict:
    """Recursively convert dataclasses to plain dicts for JSON serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = dataclasses.asdict(obj)
        # Convert enum values to their string value
        for k, v in d.items():
            if hasattr(v, 'value'):
                d[k] = v.value
        return d
    return obj

@app.get("/shap", response_class=HTMLResponse)
async def shap_get(request: Request):
    """
    SHAP Explainability Dashboard.
    In production: load text from the last uploaded paper stored in session.
    Demo mode: uses sample text so the dashboard always renders.
    """
    if not get_current_user(request):
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)

    # In production, retrieve real text from session or DB:
    # text     = request.session.get("last_paper_text", "")
    # filename = request.session.get("last_paper_name", "paper.pdf")

    # Demo: use sample AI-like text so dashboard always works
    filename = request.session.get("last_paper_name", "sample_paper.pdf")
    text = (
        "In this paper, we present a novel and comprehensive framework for addressing "
        "the critical challenge of machine learning in resource-constrained environments. "
        "Our approach leverages state-of-the-art techniques to achieve significant improvements "
        "over existing baselines, demonstrating the effectiveness and efficiency of the proposed "
        "methodology. Extensive experiments on multiple benchmark datasets confirm that our method "
        "outperforms all competing approaches by a substantial margin. We believe this work makes "
        "significant contributions to the field and opens up many promising directions for future research. "
    ) * 8  # repeat to simulate a full paper

    result: ShapResult = compute_shap_explanation(text, filename)

    # Serialise to JSON for the Jinja2 → JS bridge
    result_dict = dataclasses.asdict(result)
    result_json = json.dumps(result_dict)

    return render(request, "shap.html",
                  result=result,
                  result_json=result_json,
                  filename=filename)


# ── Adversarial Test Suite ─────────────────────────────────────

@app.get("/adversarial", response_class=HTMLResponse)
async def adversarial_get(request: Request):
    """
    Adversarial Test Suite.
    Runs all attack vectors against the last analysed paper.
    """
    if not get_current_user(request):
        flash_msg(request, "Please log in first.", "error")
        return RedirectResponse("/login", status_code=303)

    # Check if a paper has been analysed (stored in session)
    filename   = request.session.get("last_paper_name")
    verdict    = request.session.get("last_verdict")
    confidence = request.session.get("last_confidence")

    if not filename:
        # No paper in session — show upload prompt
        return render(request, "adversarial.html", report=None)

    # Demo: use the session values or sensible defaults
    report: AdversarialReport = run_adversarial_tests(
        text                = "",                 # real text would come from session/DB
        filename            = filename,
        original_verdict    = verdict    or "AI-Generated",
        original_confidence = float(confidence or 94.7),
    )

    return render(request, "adversarial.html", report=report)


@app.post("/adversarial", response_class=HTMLResponse)
async def adversarial_post(
    request:    Request,
    paper_text: str   = Form(default=""),
    filename:   str   = Form(default="paper.pdf"),
    verdict:    str   = Form(default="AI-Generated"),
    confidence: float = Form(default=94.7),
):
    """
    POST endpoint — call this from your upload/results flow to trigger
    adversarial testing with the actual paper text and result.
    Stores the result in session for the GET to render.
    """
    if not get_current_user(request):
        return RedirectResponse("/login", status_code=303)

    # Persist to session for subsequent GET
    request.session["last_paper_name"]  = filename
    request.session["last_verdict"]     = verdict
    request.session["last_confidence"]  = str(confidence)

    report: AdversarialReport = run_adversarial_tests(
        text                = paper_text,
        filename            = filename,
        original_verdict    = verdict,
        original_confidence = confidence,
    )

    return render(request, "adversarial.html", report=report)