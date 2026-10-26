"""Tracking console: Streamlit front end for object detection and tracking.

Run with:  streamlit run app.py

Requires detector.py, tracker.py and utils.py in the same folder
(ObjectDetector, ObjectTracker, Visualizer).

First-run accounts (change the passwords in Settings or Admin Panel):
    admin / Admin@123
    user  / User@123
"""

from __future__ import annotations

import os

# Must be set before cv2 is imported. Forces TCP for RTSP streams (more reliable).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import datetime as dt
import hashlib
import hmac
import html
import secrets
import sqlite3
import tempfile
import time
from collections import Counter, deque
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
from streamlit_option_menu import option_menu

try:  # torch is only used to report the compute device
    import torch
except Exception:  # pragma: no cover
    torch = None

from detector import ObjectDetector
from tracker import ObjectTracker
from utils import Visualizer

st.set_page_config(
    page_title="Tracking Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
DB_PATH = os.path.join(DATA_DIR, "app.db")

ACCENT = "#D95A1A"
ACCENT_HOVER = "#C94B1A"
COPPER = "#B85A2B"
TEXT = "#F5F0EB"
MUTED = "#A39A92"
BORDER = "#373230"
GREY = "#8A7F78"

COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
PRESET_PEOPLE: List[str] = ["person"]
PRESET_VEHICLES: List[str] = ["bicycle", "car", "motorcycle", "bus", "truck"]

SOURCES: List[str] = ["Webcam", "Upload file", "Network stream"]
PRESETS: List[str] = ["Default", "People", "Vehicles", "All"]
MODEL_FILES: Dict[str, str] = {"Fast": "yolov8n.pt", "Accurate": "yolov8s.pt"}
VIDEO_TYPES: List[str] = ["mp4", "avi", "mov", "mkv"]
MAX_RECORDS = 100_000

PAGES: List[Tuple[str, str]] = [
    ("Home", "house"),
    ("Live Monitor", "camera-video"),
    ("Video Analysis", "film"),
    ("Comparison Studio", "columns-gap"),
    ("Analytics", "bar-chart-line"),
    ("Reports", "file-earmark-text"),
    ("Settings", "gear"),
    ("Documentation", "book"),
    ("Admin Panel", "shield-lock"),
]

CONFIG_TOML = """[theme]
base = "dark"
primaryColor = "#D95A1A"
backgroundColor = "#141210"
secondaryBackgroundColor = "#24201E"
textColor = "#F5F0EB"
font = "sans serif"

[server]
maxUploadSize = 500

[client]
toolbarMode = "minimal"

[browser]
gatherUsageStats = false
"""


def ensure_streamlit_config() -> None:
    """Create .streamlit/config.toml on first run (theme applies after a restart)."""
    try:
        path = os.path.join(".streamlit", "config.toml")
        if not os.path.exists(path):
            os.makedirs(".streamlit", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(CONFIG_TOML)
    except OSError:
        pass


# =============================================================================
# 2. STYLES
# =============================================================================

# Universal icons drawn with CSS masks, so they never depend on Streamlit's icon font.
_SVG_HEAD = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
    "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
)
EYE_ICON = (
    'url("data:image/svg+xml;utf8,' + _SVG_HEAD
    + "<path d='M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z'/>"
    "<circle cx='12' cy='12' r='3'/></svg>\")"
)
EYE_OFF_ICON = (
    'url("data:image/svg+xml;utf8,' + _SVG_HEAD
    + "<path d='M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 "
    "5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 "
    "3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24'/><line x1='1' y1='1' x2='23' y2='23'/></svg>\")"
)
UPLOAD_ICON = (
    'url("data:image/svg+xml;utf8,' + _SVG_HEAD
    + "<path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/>"
    "<polyline points='17 8 12 3 7 8'/><line x1='12' y1='3' x2='12' y2='15'/></svg>\")"
)

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:#141210; --bar:#1C1917; --card:#24201E; --border:#373230;
  --text:#F5F0EB; --muted:#A39A92;
  --accent:#D95A1A; --accent-hover:#C94B1A; --copper:#B85A2B; --on-accent:#14110F;
  --ok:#3FB950; --warn:#E3A008; --bad:#F85149;
}

/* Fonts: applied to text elements only. Icon spans are never touched. */
html, body, .stApp, p, label, li, h1, h2, h3, h4, h5, h6, input, textarea, button,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"], [data-baseweb="tab"], [data-baseweb="select"] div {
  font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif !important;
}
span[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
}

/* Chrome removal */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

.stApp { background: var(--bg) !important; color: var(--text) !important; }
.block-container { max-width: 1440px !important; padding: 1.5rem 2rem 4rem !important; }

[data-testid="stSidebar"] { background: var(--bar) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }

/* Type scale */
.page-title { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; color: var(--text); }
.section-title { font-size: 18px; font-weight: 700; color: var(--text); margin: 0 0 12px; }
.greeting { font-size: 22px; font-weight: 700; letter-spacing: -0.01em; margin: 4px 0 20px; }
.greeting b { color: var(--accent); font-weight: 800; }
.muted { color: var(--muted); font-size: 13px; }
.gap { height: 24px; }

/* Top bar */
div.st-key-topbar .stButton button {
    width: auto !important;             /* Button ko stretch hone se rokega */
    padding: 4px 10px !important;       /* Button ke andar ki padding kam karega */
    min-height: 32px !important;        /* Height ko compact karega */
    font-size: 13px !important;
}

/* Columns extra margin/gap */
div.st-key-topbar div[data-testid="column"] {
    padding-left: 2px !important;
    padding-right: 2px !important;
}

/* Cards */
div[class*="st-key-card_"] {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
div[class*="st-key-card_accent"] { border-left: 3px solid var(--accent); }

.kpi {
  background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 10px; padding: 20px 24px;
}
.kpi-label { font-size: 13px; font-weight: 500; color: var(--muted); }
.kpi-value { font-size: 32px; font-weight: 800; letter-spacing: -0.02em; color: var(--accent); line-height: 1.15; margin-top: 6px; }

.mini-row { display: flex; gap: 12px; margin: 12px 0 16px; }
.mini { flex: 1; background: var(--bar); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.mini .kpi-value { font-size: 22px; margin-top: 2px; }

.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
.chip { border: 1px solid var(--border); background: var(--card); border-radius: 999px; padding: 6px 14px; font-size: 13px; font-weight: 600; }

.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
  min-height: 420px; background: var(--card); border: 1.5px dashed var(--border); border-radius: 10px;
  color: var(--muted); font-size: 16px; font-weight: 600;
}
.empty-state svg { color: var(--accent); }
.note { background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 14px 18px; color: var(--muted); font-size: 14px; }

.qa-icon { color: var(--accent); margin-bottom: 12px; }
.qa-title { font-size: 20px; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 16px; }

/* Buttons */
.stButton button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {
  width: 100%; min-height: 44px;
}
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"], button[kind="primary"] {
  background: var(--accent) !important; border: 1px solid var(--accent) !important;
  color: var(--on-accent) !important; border-radius: 8px !important; font-weight: 700 !important;
}
button[data-testid="stBaseButton-primary"] *, button[data-testid="stBaseButton-primaryFormSubmit"] *, button[kind="primary"] * {
  color: var(--on-accent) !important; font-weight: 700 !important;
}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover, button[kind="primary"]:hover {
  background: var(--accent-hover) !important; border-color: var(--accent-hover) !important;
  box-shadow: 0 0 0 3px rgba(217,90,26,0.25) !important;
}
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-secondaryFormSubmit"], button[kind="secondary"] {
  background: #2E2927 !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 8px !important; font-weight: 600 !important;
}
button[data-testid="stBaseButton-secondary"] *, button[data-testid="stBaseButton-secondaryFormSubmit"] *, button[kind="secondary"] * {
  color: var(--text) !important;
}
button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-secondaryFormSubmit"]:hover, button[kind="secondary"]:hover {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(217,90,26,0.25) !important;
}
button[data-testid="stBaseButton-secondary"]:hover *, button[kind="secondary"]:hover * { color: var(--accent) !important; }

/* Inputs */

[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"],
div[data-baseweb="input"] {
    background-color: #171513 !important;
    border: 1px solid #4D4440 !important;
    border-radius: 8px !important;
    padding: 3px 8px !important;
    margin-top: 4px !important;
}

[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stNumberInput"] div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="input"]:focus-within {
    border-color: #D95A1A !important;
    box-shadow: 0 0 0 2px rgba(217, 90, 26, 0.25) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    color: #F5F0EB !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 6px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #171513 !important;
    border: 1px solid #4D4440 !important;
    border-radius: 8px !important;
    margin-top: 4px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: #D95A1A !important;
    box-shadow: 0 0 0 2px rgba(217, 90, 26, 0.25) !important;
}

/* Tabs */
button[data-baseweb="tab"] { color: var(--muted) !important; font-weight: 600 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 3px !important; }
[data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* Panels */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 10px !important; background: var(--card); }
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--border); }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
[data-testid="stProgress"] > div > div > div > div { background-color: var(--accent) !important; }
[data-testid="stImage"] img { border-radius: 10px; }

/* File uploader: one clean button, universal upload icon */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important; border: 1.5px dashed var(--border) !important;
  border-radius: 10px !important; padding: 24px !important; transition: border-color .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] svg { display: none !important; }
[data-testid="stFileUploaderDropzone"] button {
  display: inline-flex !important; align-items: center; justify-content: center; gap: 8px;
}
[data-testid="stFileUploaderDropzone"] button::before {
  content: ""; width: 16px; height: 16px; background-color: currentColor; display: inline-block;
  -webkit-mask: __UPLOAD__ center / contain no-repeat; mask: __UPLOAD__ center / contain no-repeat;
}
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] * {
  color: var(--muted) !important; font-size: 13px !important;
}

/* Password field: native glyph hidden, universal eye icon drawn instead */
[data-testid="stTextInput"] button {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  min-height: 0 !important; width: auto !important; padding: 0 10px !important;
}
[data-testid="stTextInput"] button > * { display: none !important; }
[data-testid="stTextInput"] button::after {
  content: ""; display: block; width: 18px; height: 18px; background-color: var(--muted);
  -webkit-mask: __EYE__ center / contain no-repeat; mask: __EYE__ center / contain no-repeat;
}
[data-testid="stTextInput"] button:hover::after { background-color: var(--accent); }
[data-testid="stTextInput"] button[aria-label*="Hide"]::after, [data-testid="stTextInput"] button[title*="Hide"]::after {
  -webkit-mask: __EYE_OFF__ center / contain no-repeat; mask: __EYE_OFF__ center / contain no-repeat;
}
[data-testid="stSidebar"] { 
    display: block !important; 
    visibility: visible !important; 
    background: var(--bar) !important; 
    border-right: 1px solid var(--border); 
}
[data-testid="stSidebarCollapsedControl"] { 
    display: flex !important; 
    visibility: visible !important; 
}
</style>
"""

LOGIN_CSS = """
<style>
.stApp {
  background-color: #141210 !important;
  background-image:
    radial-gradient(900px 560px at 6% -4%, rgba(217,90,26,0.30), transparent 62%),
    radial-gradient(720px 480px at 100% 110%, rgba(184,90,43,0.18), transparent 60%),
    linear-gradient(rgba(245,240,235,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245,240,235,0.028) 1px, transparent 1px) !important;
  background-size: auto, auto, 48px 48px, 48px 48px !important;
  background-attachment: fixed !important;
}
.stApp::before, .stApp::after {
  content: ""; position: fixed; width: 64px; height: 64px; pointer-events: none; z-index: 0;
}
.stApp::before { top: 28px; left: 28px; border-top: 2px solid #D95A1A; border-left: 2px solid #D95A1A; }
.stApp::after { bottom: 28px; right: 28px; border-bottom: 2px solid #D95A1A; border-right: 2px solid #D95A1A; }

div.st-key-login_card {
  width: 420px; max-width: 100%; margin: 12vh auto 0; padding: 32px 32px 28px;
  background: var(--card); border: 1px solid var(--border); border-top: 3px solid var(--accent);
  border-radius: 10px; box-shadow: 0 24px 64px rgba(0,0,0,0.55); position: relative; z-index: 1;
}
.login-mark { width: 36px; height: 36px; border-radius: 8px; background: var(--accent); position: relative; margin-bottom: 20px; }
.login-mark::after { content: ""; position: absolute; inset: 10px; border-radius: 3px; background: var(--on-accent); }
.login-title { font-size: 28px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 20px; }
</style>
"""


def build_css() -> str:
    """Return the base stylesheet with icon placeholders resolved."""
    return (
        BASE_CSS.replace("__EYE_OFF__", EYE_OFF_ICON)
        .replace("__EYE__", EYE_ICON)
        .replace("__UPLOAD__", UPLOAD_ICON)
    )


NAV_STYLES: Dict[str, Dict[str, str]] = {
    "container": {"padding": "8px", "background-color": "#1C1917", "border-radius": "10px"},
    "icon": {"color": "inherit", "font-size": "16px"},
    "nav-link": {
        "font-size": "14px", "font-weight": "500", "color": "#A39A92", "text-align": "left",
        "margin": "4px 0", "padding": "12px 14px", "border-radius": "8px",
        "--hover-color": "rgba(217,90,26,0.14)",
    },
    "nav-link-selected": {"background-color": "#D95A1A", "color": "#14110F", "font-weight": "700"},
}

_ICON_PATHS: Dict[str, str] = {
    "video": "<polygon points='23 7 16 12 23 17 23 7'/><rect x='1' y='5' width='15' height='14' rx='2' ry='2'/>",
    "film": (
        "<rect x='2' y='2' width='20' height='20' rx='2.18' ry='2.18'/><line x1='7' y1='2' x2='7' y2='22'/>"
        "<line x1='17' y1='2' x2='17' y2='22'/><line x1='2' y1='12' x2='22' y2='12'/>"
        "<line x1='2' y1='7' x2='7' y2='7'/><line x1='2' y1='17' x2='7' y2='17'/>"
        "<line x1='17' y1='17' x2='22' y2='17'/><line x1='17' y1='7' x2='22' y2='7'/>"
    ),
    "columns": "<path d='M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7m0-18H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7m0-18v18'/>",
    "target": "<circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/>",
}


def icon_svg(name: str, size: int = 28) -> str:
    """Return an inline Feather-style SVG that inherits the current text color."""
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='{size}' height='{size}' "
        "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        f"{_ICON_PATHS.get(name, '')}</svg>"
    )


# =============================================================================
# 3. DATABASE AND AUTHENTICATION
# =============================================================================

def db_exec(sql: str, params: Sequence[Any] = ()) -> int:
    """Run a write statement and return the last row id."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return int(cur.lastrowid or 0)


def db_rows(sql: str, params: Sequence[Any] = ()) -> List[Tuple[Any, ...]]:
    """Run a read statement and return all rows."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return list(conn.execute(sql, tuple(params)).fetchall())


def db_df(sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    """Run a read statement and return a DataFrame."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(sql, conn, params=tuple(params))


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Return (hash, salt) using salted PBKDF2-SHA256."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120_000)
    return digest.hex(), salt


@st.cache_resource
def init_database() -> bool:
    """Create folders, tables and the two seed accounts (runs once per server start)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                model TEXT, conf REAL, classes TEXT, width INTEGER
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, name TEXT, source TEXT,
                started_at TEXT, ended_at TEXT,
                frames INTEGER DEFAULT 0, objects INTEGER DEFAULT 0, avg_fps REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS session_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER, track_id INTEGER, class_name TEXT
            );
            CREATE TABLE IF NOT EXISTS session_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER, class_name TEXT, cx REAL, cy REAL
            );
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT, action TEXT, detail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            for name, pwd, role in (("admin", "Admin@123", "Admin"), ("user", "User@123", "User")):
                pwd_hash, salt = hash_password(pwd)
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                    (name, pwd_hash, salt, role),
                )
        conn.commit()
    return True


def log_activity(username: str, action: str, detail: str = "") -> None:
    """Append an entry to the activity log."""
    db_exec("INSERT INTO activity_logs (username, action, detail) VALUES (?, ?, ?)", (username, action, detail))


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Return the user dict if the credentials are valid and the account is active."""
    rows = db_rows(
        "SELECT id, username, password_hash, salt, role, is_active FROM users WHERE username = ?",
        (username,),
    )
    if not rows:
        return None
    uid, uname, stored, salt, role, active = rows[0]
    if not active:
        return None
    candidate, _ = hash_password(password, salt)
    if hmac.compare_digest(candidate, stored):
        return {"id": uid, "username": uname, "role": role}
    return None


def set_password(user_id: int, new_password: str) -> None:
    """Replace a user's password."""
    pwd_hash, salt = hash_password(new_password)
    db_exec("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?", (pwd_hash, salt, user_id))


DEFAULT_PREFS: Dict[str, Any] = {"model": "Fast", "conf": 0.40, "classes": ["person", "car"], "width": 640}


def get_user_settings(user_id: int) -> Dict[str, Any]:
    """Return saved defaults for a user, falling back to built-ins."""
    rows = db_rows("SELECT model, conf, classes, width FROM user_settings WHERE user_id = ?", (user_id,))
    if not rows:
        return dict(DEFAULT_PREFS)
    model, conf, classes, width = rows[0]
    names = [c for c in str(classes or "").split(",") if c in COCO_CLASSES]
    return {
        "model": model if model in MODEL_FILES else "Fast",
        "conf": float(conf or 0.40),
        "classes": names or ["person"],
        "width": int(width or 640),
    }


def save_user_settings(user_id: int, prefs: Dict[str, Any]) -> None:
    """Persist per-user defaults."""
    db_exec(
        "INSERT OR REPLACE INTO user_settings (user_id, model, conf, classes, width) VALUES (?, ?, ?, ?, ?)",
        (user_id, prefs["model"], float(prefs["conf"]), ",".join(prefs["classes"]), int(prefs["width"])),
    )


def scope_clause(alias: str = "s") -> Tuple[str, Tuple[Any, ...]]:
    """Admins see all sessions; users see their own."""
    user = st.session_state.get("auth_user", {})
    if user.get("role") == "Admin":
        return "1=1", ()
    return f"{alias}.user_id = ?", (user.get("id", -1),)


# =============================================================================
# 4. UI HELPERS
# =============================================================================

def keyed_container(key: str) -> Any:
    """Container with a CSS hook (class st-key-<key>); falls back on old Streamlit."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def make_columns(spec: Any, **kwargs: Any) -> Any:
    """st.columns with vertical centering when supported."""
    try:
        return st.columns(spec, vertical_alignment="center", **kwargs)
    except TypeError:
        return st.columns(spec, **kwargs)


def kpi_html(label: str, value: Any) -> str:
    """Return the HTML of one KPI card."""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'


def mini_stats(items: Sequence[Tuple[str, Any]]) -> str:
    """Return a compact row of small stat tiles."""
    tiles = "".join(
        f'<div class="mini"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="mini-row">{tiles}</div>'


def empty_state(message: str) -> None:
    """Render a quiet one-line empty notice."""
    st.markdown(f'<div class="note">{html.escape(message)}</div>', unsafe_allow_html=True)


def spacer() -> None:
    """Vertical gap between sections."""
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)


def show_image(target: Any, image: Any) -> None:
    """Render an image stretched to its container on old and new Streamlit."""
    try:
        target.image(image, width="stretch")
    except Exception:
        target.image(image, use_container_width=True)


def show_dataframe(df: pd.DataFrame, target: Any = None) -> None:
    """Render a table stretched to its container on old and new Streamlit."""
    out = target if target is not None else st
    try:
        out.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        out.dataframe(df, use_container_width=True, hide_index=True)


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    """Apply the dark transparent chart theme."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=TEXT, size=12),
        margin=dict(l=8, r=8, t=24, b=8),
        height=height,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def show_chart(fig: go.Figure, target: Any = None, key: Optional[str] = None) -> None:
    """Render a Plotly chart without Streamlit overriding the theme."""
    out = target if target is not None else st
    try:
        out.plotly_chart(fig, theme=None, width="stretch", key=key)
    except Exception:
        out.plotly_chart(fig, theme=None, use_container_width=True, key=key)


def keep_state(keys: Sequence[str]) -> None:
    """Preserve widget values while the person visits other pages."""
    for key in keys:
        if key in st.session_state:
            try:
                st.session_state[key] = st.session_state[key]
            except Exception:
                pass


def go_to(page: str) -> None:
    """Switch page and rebuild the sidebar menu at the new position."""
    st.session_state["page"] = page
    st.session_state["nav_nonce"] = st.session_state.get("nav_nonce", 0) + 1
    st.rerun()


def device_label() -> str:
    """Return a human-readable compute device."""
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "GPU (CUDA)"
        except Exception:
            pass
    return "CPU"


def classes_for(preset: str, saved: Sequence[str]) -> List[str]:
    """Resolve a class preset to a class-name list."""
    if preset == "People":
        return list(PRESET_PEOPLE)
    if preset == "Vehicles":
        return list(PRESET_VEHICLES)
    if preset == "Default" and saved:
        return list(saved)
    return list(COCO_CLASSES)


# =============================================================================
# 5. VIDEO PIPELINE
# =============================================================================

@dataclass
class RunStats:
    """Mutable results collected while a stream is processed."""

    frames: int = 0
    unique: Dict[int, str] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    points: List[Tuple[float, float, str]] = field(default_factory=list)
    fps_sum: float = 0.0
    fps_n: int = 0
    elapsed: float = 0.0
    snapshot: Any = None
    codec: str = ""

    @property
    def avg_fps(self) -> float:
        """Mean of the smoothed per-frame FPS."""
        return self.fps_sum / self.fps_n if self.fps_n else 0.0

    def records_df(self) -> pd.DataFrame:
        """Detections as a DataFrame."""
        cols = ["Frame", "Track ID", "Class", "Left", "Top", "Right", "Bottom"]
        return pd.DataFrame(self.records, columns=cols)


@st.cache_resource(show_spinner="Loading model")
def get_detector(model_file: str, conf: float) -> ObjectDetector:
    """Create and cache a detector for a model file and confidence."""
    return ObjectDetector(model_path=model_file, conf_thresh=float(conf))


def prepare_frame(frame: Any, width: int) -> Any:
    """Downscale to the target width and keep even dimensions (needed by video codecs)."""
    h, w = frame.shape[:2]
    if w > width:
        scale = width / float(w)
        frame = cv2.resize(frame, (int(width) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
    else:
        frame = frame[: h // 2 * 2, : w // 2 * 2]
    return frame


def open_writer(path: str, fps: float, size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], str]:
    """Open a video writer, preferring a browser-playable codec."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, ""


def new_output_path(prefix: str) -> str:
    """Return a unique output video path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}.mp4")


def save_upload(uploaded: Any) -> str:
    """Write an uploaded file to a temp path once and reuse it."""
    cache: Dict[str, str] = st.session_state.setdefault("_upload_cache", {})
    key = f"{uploaded.name}:{uploaded.size}"
    path = cache.get(key)
    if path and os.path.exists(path):
        return path
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getbuffer())
        cache[key] = handle.name
    return cache[key]


def read_meta(path: str) -> Optional[Dict[str, Any]]:
    """Return basic video metadata, or None if the file cannot be opened."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": fps,
            "frames": frames,
            "duration": frames / fps if fps > 0 else 0.0,
        }
    finally:
        cap.release()


def build_stream_url(url: str, user: str, password: str) -> str:
    """Insert optional credentials into a stream URL (URL-encoded)."""
    url = url.strip()
    if not user and not password:
        return url
    parts = urlsplit(url)
    if "@" in parts.netloc or not parts.netloc:
        return url
    auth = quote(user, safe="") + (":" + quote(password, safe="") if password else "")
    return urlunsplit((parts.scheme, f"{auth}@{parts.netloc}", parts.path, parts.query, parts.fragment))


def resolve_stream(url: str) -> str:
    """Return a URL OpenCV can open (works for both normal videos and YouTube Live streams)."""
    lowered = url.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        # Method 1: Streamlink (Live streams ke liye 100% reliable)
        try:
            import streamlink
            session = streamlink.Streamlink()
            session.set_option("hls-live-edge", 2)
            streams = session.streams(url)
            if "best" in streams:
                return str(streams["best"].url)
            if "720p" in streams:
                return str(streams["720p"].url)
            if streams:
                return str(list(streams.values())[0].url)
        except Exception:
            pass

        # Method 2: yt-dlp with mobile client simulation
        try:
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "format": "best[protocol^=m3u8]/bestvideo+bestaudio/best/worst",
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "ios", "web"]
                    }
                },
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if "url" in info:
                    return str(info["url"])

                # Formats list me se live HLS stream nikalna
                formats = info.get("formats", [])
                for f in reversed(formats):
                    f_url = f.get("url", "")
                    protocol = f.get("protocol", "")
                    if f_url and ("m3u8" in protocol or ".m3u8" in f_url):
                        return str(f_url)
                if formats:
                    return str(formats[-1].get("url"))
        except Exception as exc:
            raise RuntimeError(f"Could not extract stream: {exc}") from exc

    return url


def open_capture(cfg: Dict[str, Any]) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Open the configured source. Returns (capture, error message)."""
    source = cfg["source"]
    if source == "Webcam":
        cap = cv2.VideoCapture(int(cfg["cam"]))
        error = "Camera not available. Check the index and close other apps using it."
    elif source == "Upload file":
        if cfg["file"] is None:
            return None, "Upload a video file first."
        cap = cv2.VideoCapture(save_upload(cfg["file"]))
        error = "This video file could not be opened."
    else:
        url = str(cfg["url"]).strip()
        if not url:
            return None, "Enter a stream address first."
        try:
            target = resolve_stream(build_stream_url(url, str(cfg["user"]), str(cfg["pwd"])))
        except Exception as exc:
            return None, f"Could not resolve the stream: {exc}"
        cap = cv2.VideoCapture(target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        error = "Could not open the stream. Check the address, credentials and network."
    if not cap.isOpened():
        cap.release()
        return None, error
    return cap, ""


def run_pipeline(
    cap: cv2.VideoCapture,
    detector: Any,
    tracker: Any,
    vis: Any,
    stats: RunStats,
    *,
    classes: Sequence[str],
    width: int,
    show_fps: bool,
    out_path: Optional[str] = None,
    on_frame: Optional[Callable[[Any, List[Tuple[Any, Any, Any]], float], None]] = None,
    max_frames: Optional[int] = None,
) -> None:
    """Detect, track and annotate frames until the stream ends.

    Keeps the original logic: detector.detect(frame, allowed_classes=...),
    tracker.update(detections, frame) -> (track_id, ltrb, class_name), and a
    2-frame hit counter before a track is accepted.
    """
    hits: Dict[int, int] = {}
    window: Deque[float] = deque(maxlen=15)
    writer: Optional[cv2.VideoWriter] = None
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not (1.0 <= src_fps <= 120.0):
        src_fps = 25.0
    began = time.perf_counter()
    try:
        while max_frames is None or stats.frames < max_frames:
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            stats.frames += 1
            frame = prepare_frame(frame, width)
            h, w = frame.shape[:2]

            detections = detector.detect(frame, allowed_classes=list(classes))
            tracks = tracker.update(detections, frame)

            valid: List[Tuple[Any, Any, Any]] = []
            for track_id, ltrb, cls_name in tracks:
                tid = int(track_id)
                hits[tid] = hits.get(tid, 0) + 1
                if hits[tid] < 2:
                    continue
                valid.append((track_id, ltrb, cls_name))
                label = str(cls_name)
                stats.unique.setdefault(tid, label)
                try:
                    left, top, right, bottom = (float(v) for v in ltrb)
                except (TypeError, ValueError):
                    left = top = right = bottom = 0.0
                if len(stats.records) < MAX_RECORDS:
                    stats.records.append({
                        "Frame": stats.frames, "Track ID": tid, "Class": label,
                        "Left": round(left), "Top": round(top), "Right": round(right), "Bottom": round(bottom),
                    })
                if stats.frames % 5 == 0 and w and h:
                    stats.points.append((((left + right) / 2) / w, ((top + bottom) / 2) / h, label))

            vis.draw_tracks(frame, valid)
            if show_fps:
                vis.draw_fps(frame)

            if out_path:
                if writer is None:
                    writer, stats.codec = open_writer(out_path, src_fps, (w, h))
                    if writer is None:
                        out_path = None
                if writer is not None:
                    writer.write(frame)
            if valid and (stats.snapshot is None or stats.frames % 10 == 0):
                stats.snapshot = frame.copy()

            spent = time.perf_counter() - tick
            window.append(1.0 / spent if spent > 0 else 0.0)
            fps = sum(window) / len(window)
            stats.fps_sum += fps
            stats.fps_n += 1
            if on_frame is not None:
                on_frame(frame, valid, fps)
    finally:
        if writer is not None:
            writer.release()
        stats.elapsed = time.perf_counter() - began


def persist_session(stats: RunStats, user_id: int, prefix: str, source: str, started: dt.datetime) -> int:
    """Save a finished run to the database and return its id."""
    if stats.frames == 0:
        return 0
    ended = dt.datetime.now()
    sid = db_exec(
        "INSERT INTO sessions (user_id, name, source, started_at, ended_at, frames, objects, avg_fps) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id, f"{prefix} {started:%Y-%m-%d %H:%M}", source,
            started.strftime("%Y-%m-%d %H:%M:%S"), ended.strftime("%Y-%m-%d %H:%M:%S"),
            stats.frames, len(stats.unique), round(stats.avg_fps, 2),
        ),
    )
    step = max(1, len(stats.points) // 5000)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executemany(
            "INSERT INTO session_tracks (session_id, track_id, class_name) VALUES (?, ?, ?)",
            [(sid, tid, cls) for tid, cls in stats.unique.items()],
        )
        conn.executemany(
            "INSERT INTO session_points (session_id, class_name, cx, cy) VALUES (?, ?, ?, ?)",
            [(sid, cls, cx, cy) for cx, cy, cls in stats.points[::step]],
        )
        conn.commit()
    return sid


def counts_figure(unique: Dict[int, str]) -> go.Figure:
    """Bar chart of unique objects per class."""
    tally = Counter(unique.values())
    df = pd.DataFrame({"Class": list(tally.keys()), "Objects": list(tally.values())})
    fig = px.bar(df, x="Class", y="Objects")
    fig.update_traces(marker_color=ACCENT)
    return style_fig(fig, 240)


# =============================================================================
# 6. PAGES: SIGN IN AND TOP BAR
# =============================================================================

def render_login() -> None:
    """Sign-in screen."""
    with keyed_container("login_card"):
        st.markdown('<div class="login-mark"></div><div class="login-title">Sign in</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            user = authenticate(username.strip(), password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["page"] = "Home"
                log_activity(user["username"], "Sign in")
                st.rerun()
            else:
                st.error("Incorrect username or password.")


def render_top_bar(title: str) -> None:
    """Page title with compact top navigation buttons and user details."""
    user = st.session_state["auth_user"]
    with keyed_container("topbar"):
        left, right = make_columns([6.5, 3.5])
        with left:
            # gap="small" aur tight column ratios se buttons ekdum paas aa jayenge
            t_cols = st.columns([0.50, 0.50, 0.60, 0.60, 4.5], gap="small")
            with t_cols[0]:
                if st.button("Home", key="top_nav_home"):
                    go_to("Home")
            with t_cols[1]:
                if st.button("Live", key="top_nav_live"):
                    go_to("Live Monitor")
            with t_cols[2]:
                if st.button("Settings", key="top_nav_settings"):
                    go_to("Settings")
            with t_cols[3]:
                if user["role"] == "Admin":
                    if st.button("Admin Panel", key="top_nav_admin"):
                        go_to("Admin Panel")

        with right:
            info, action = make_columns([3, 1.3])
            info.markdown(
                f'<div class="user-line"><span>{html.escape(user["username"])}</span>'
                f'<span class="role-pill">{html.escape(user["role"])}</span></div>',
                unsafe_allow_html=True,
            )
            if action.button("Sign out", key="signout"):
                log_activity(user["username"], "Sign out")
                st.session_state.clear()
                st.rerun()


# =============================================================================
# 7. PAGES: HOME
# =============================================================================

def render_home() -> None:
    """Greeting, three KPIs, quick actions and recent sessions."""
    user = st.session_state["auth_user"]
    clause, params = scope_clause("s")
    row = db_rows(
        f"SELECT COUNT(*), COALESCE(SUM(s.objects), 0), COALESCE(AVG(s.avg_fps), 0) FROM sessions s WHERE {clause}",
        params,
    )[0]

    st.markdown(f'<div class="greeting">Welcome back, <b>{html.escape(user["username"])}</b></div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    cols[0].markdown(kpi_html("Sessions", int(row[0])), unsafe_allow_html=True)
    cols[1].markdown(kpi_html("Objects tracked", int(row[1])), unsafe_allow_html=True)
    cols[2].markdown(kpi_html("Average FPS", f"{float(row[2]):.1f}"), unsafe_allow_html=True)

    spacer()
    actions = [("video", "Live Monitor"), ("film", "Video Analysis"), ("columns", "Comparison Studio")]
    cols = st.columns(3, gap="large")
    for col, (icon, target) in zip(cols, actions):
        with col:
            with keyed_container(f"card_qa_{icon}"):
                st.markdown(
                    f'<div class="qa-icon">{icon_svg(icon, 30)}</div><div class="qa-title">{target}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"qa_{icon}", type="primary"):
                    go_to(target)

    spacer()
    st.markdown('<div class="section-title">Recent sessions</div>', unsafe_allow_html=True)
    recent = db_df(
        "SELECT s.name AS Session, s.source AS Source, s.started_at AS Started, s.objects AS Objects, "
        f"ROUND(s.avg_fps, 1) AS FPS FROM sessions s WHERE {clause} ORDER BY s.id DESC LIMIT 5",
        params,
    )
    if recent.empty:
        empty_state("No sessions yet. Open Live Monitor to start one.")
    else:
        show_dataframe(recent)


# =============================================================================
# 8. PAGES: LIVE MONITOR
# =============================================================================

LM_KEYS = [
    "lm_source", "lm_cam", "lm_url", "lm_user", "lm_pass", "lm_conf", "lm_preset",
    "lm_model", "lm_width", "lm_maxage", "lm_overlay", "lm_save", "lm_custom",
]
READY_HTML = f'<div class="empty-state">{icon_svg("target", 40)}<div>Ready</div></div>'


def ensure_lm_defaults() -> None:
    """Seed Live Monitor widget state from the user's saved defaults."""
    state = st.session_state
    if "lm_conf" in state:
        return
    prefs = get_user_settings(state["auth_user"]["id"])
    state.update({
        "lm_source": SOURCES[0], "lm_cam": 0, "lm_url": "", "lm_user": "", "lm_pass": "",
        "lm_conf": float(prefs["conf"]), "lm_preset": "Default", "lm_model": prefs["model"],
        "lm_width": int(prefs["width"]), "lm_maxage": 30, "lm_overlay": True, "lm_save": False,
        "lm_custom": list(prefs["classes"]),
    })


def render_live_monitor() -> None:
    """Live detection and tracking console."""
    keep_state(LM_KEYS)
    ensure_lm_defaults()
    state = st.session_state
    state.setdefault("is_tracking", False)

    left, right = st.columns([7, 3], gap="large")

    with right:
        with keyed_container("card_controls"):
            st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)
            source = st.selectbox("Source", SOURCES, key="lm_source")
            uploaded = None
            if source == "Webcam":
                st.number_input("Camera index", min_value=0, max_value=9, step=1, key="lm_cam")
            elif source == "Upload file":
                uploaded = st.file_uploader("Video file", type=VIDEO_TYPES, key="lm_file", label_visibility="collapsed")
            else:
                st.text_input("Stream address", key="lm_url", placeholder="rtsp:// or http://")
                c_user, c_pass = st.columns(2)
                c_user.text_input("Username", key="lm_user")
                c_pass.text_input("Password", type="password", key="lm_pass")
            st.slider("Confidence", 0.10, 0.95, step=0.05, key="lm_conf")
            st.selectbox("Classes", PRESETS, key="lm_preset")
            with st.expander("Advanced"):
                st.selectbox("Model", list(MODEL_FILES), key="lm_model")
                st.slider("Frame width", 320, 1280, step=80, key="lm_width")
                st.number_input("Track memory (frames)", min_value=5, max_value=120, step=5, key="lm_maxage")
                st.multiselect("Default classes", COCO_CLASSES, key="lm_custom")
                st.toggle("Show FPS on video", key="lm_overlay")
                st.toggle("Save annotated video", key="lm_save")
            b_start, b_stop = st.columns(2)
            if b_start.button("Start", type="primary", key="lm_start"):
                state["is_tracking"] = True
            if b_stop.button("Stop", key="lm_stop"):
                state["is_tracking"] = False

    last = state.get("last_run")
    tracking = bool(state["is_tracking"])

    with left:
        error = state.pop("lm_error", None)
        if error:
            st.error(error)
        k1, k2, k3 = st.columns(3, gap="medium")
        slot_fps, slot_now, slot_uniq = k1.empty(), k2.empty(), k3.empty()
        idle_fps = f"{last['avg_fps']:.1f}" if last and not tracking else "--"
        idle_uniq = len(last["unique"]) if last and not tracking else 0
        slot_fps.markdown(kpi_html("FPS", idle_fps), unsafe_allow_html=True)
        slot_now.markdown(kpi_html("Objects", 0), unsafe_allow_html=True)
        slot_uniq.markdown(kpi_html("Unique IDs", idle_uniq), unsafe_allow_html=True)

        video_slot = st.empty()
        if not tracking:
            video_slot.markdown(READY_HTML, unsafe_allow_html=True)

        tab_log, tab_counts = st.tabs(["Log", "Counts"])
        log_slot, counts_slot = tab_log.empty(), tab_counts.empty()
        if last and not tracking:
            show_dataframe(last["log"].tail(30).iloc[::-1], log_slot)
            if last["unique"]:
                show_chart(counts_figure(last["unique"]), counts_slot, key="idle_counts_chart")
        else:
            log_slot.markdown('<div class="muted">No detections yet.</div>', unsafe_allow_html=True)

        if last and not tracking:
            d1, d2 = st.columns(2)
            d1.download_button(
                "Download CSV", last["log"].to_csv(index=False).encode("utf-8"),
                file_name="tracking_log.csv", mime="text/csv", key="dl_live_csv",
            )
            if last.get("video") and os.path.exists(last["video"]):
                with open(last["video"], "rb") as handle:
                    d2.download_button("Download video", handle.read(), file_name="tracking.mp4", mime="video/mp4", key="dl_live_video")

    if tracking:
        cfg = {
            "source": source, "cam": state["lm_cam"], "file": uploaded, "url": state["lm_url"],
            "user": state["lm_user"], "pwd": state["lm_pass"], "conf": state["lm_conf"],
            "model_file": MODEL_FILES.get(state["lm_model"], "yolov8n.pt"), "width": int(state["lm_width"]),
            "max_age": int(state["lm_maxage"]), "overlay": bool(state["lm_overlay"]), "save": bool(state["lm_save"]),
            "classes": classes_for(state["lm_preset"], state["lm_custom"]),
        }
        ui = {"video": video_slot, "fps": slot_fps, "now": slot_now, "uniq": slot_uniq, "log": log_slot, "counts": counts_slot}
        run_live(cfg, ui)


def run_live(cfg: Dict[str, Any], ui: Dict[str, Any]) -> None:
    """Process the live source and update the page until it ends or is stopped."""
    state = st.session_state
    cap, message = open_capture(cfg)
    if cap is None:
        state["is_tracking"] = False
        state["lm_error"] = message
        st.rerun()
        return
    try:
        detector = get_detector(cfg["model_file"], cfg["conf"])
        tracker = ObjectTracker(max_age=cfg["max_age"])
        vis = Visualizer()
    except Exception as exc:
        cap.release()
        state["is_tracking"] = False
        state["lm_error"] = f"The model could not be loaded: {exc}"
        st.rerun()
        return

    stats = RunStats()
    started = dt.datetime.now()
    out_path = new_output_path("live") if cfg["save"] else None
    error: Optional[str] = None

    def on_frame(frame: Any, valid: List[Tuple[Any, Any, Any]], fps: float) -> None:
        show_image(ui["video"], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if stats.frames % 3 == 1:
            ui["fps"].markdown(kpi_html("FPS", f"{fps:.1f}"), unsafe_allow_html=True)
            ui["now"].markdown(kpi_html("Objects", len(valid)), unsafe_allow_html=True)
            ui["uniq"].markdown(kpi_html("Unique IDs", len(stats.unique)), unsafe_allow_html=True)
        if stats.frames % 15 == 0 and stats.records:
            log = pd.DataFrame(stats.records[-30:][::-1])[["Frame", "Track ID", "Class"]]
            show_dataframe(log, ui["log"])
            show_chart(counts_figure(stats.unique), ui["counts"], key=f"live_chart_{stats.frames}")

    try:
        run_pipeline(
            cap, detector, tracker, vis, stats,
            classes=cfg["classes"], width=cfg["width"], show_fps=cfg["overlay"],
            out_path=out_path, on_frame=on_frame,
        )
    except Exception as exc:
        error = f"Processing stopped: {exc}"
    finally:
        cap.release()
        if stats.frames:
            persist_session(stats, state["auth_user"]["id"], "Live", cfg["source"], started)
            state["last_run"] = {
                "avg_fps": stats.avg_fps, "unique": dict(stats.unique), "log": stats.records_df(),
                "video": out_path if stats.codec else None,
            }
    state["is_tracking"] = False
    if error:
        state["lm_error"] = error
    st.rerun()


# =============================================================================
# 9. PAGES: VIDEO ANALYSIS AND COMPARISON
# =============================================================================

def render_video_analysis() -> None:
    """Offline processing of one video file."""
    keep_state(["va_model", "va_conf"])
    uid = st.session_state["auth_user"]["id"]
    prefs = get_user_settings(uid)
    st.session_state.setdefault("va_model", prefs["model"])
    st.session_state.setdefault("va_conf", float(prefs["conf"]))

    uploaded = st.file_uploader("Video file", type=VIDEO_TYPES, key="va_file", label_visibility="collapsed")
    if uploaded is None:
        st.markdown('<div class="muted">Upload a video to begin.</div>', unsafe_allow_html=True)
        return

    path = save_upload(uploaded)
    meta = read_meta(path)
    if meta is None:
        st.error("This video file could not be opened.")
        return

    st.markdown(
        '<div class="chips">'
        f'<span class="chip">{meta["width"]} x {meta["height"]}</span>'
        f'<span class="chip">{meta["fps"]:.0f} FPS</span>'
        f'<span class="chip">{meta["duration"]:.1f} s</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Options"):
        o1, o2 = st.columns(2)
        o1.selectbox("Model", list(MODEL_FILES), key="va_model")
        o2.slider("Confidence", 0.10, 0.95, step=0.05, key="va_conf")

    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.button("Process video", type="primary", key="va_run"):
        st.session_state.pop("va_result", None)
        cap = cv2.VideoCapture(path)
        stats = RunStats()
        started = dt.datetime.now()
        out_path = new_output_path("analysis")
        bar = st.progress(0.0)
        total = max(int(meta["frames"]), 1)

        def on_frame(frame: Any, valid: List[Tuple[Any, Any, Any]], fps: float) -> None:
            if stats.frames % 5 == 0:
                bar.progress(min(stats.frames / total, 1.0))

        error: Optional[str] = None
        try:
            run_pipeline(
                cap,
                get_detector(MODEL_FILES[st.session_state["va_model"]], st.session_state["va_conf"]),
                ObjectTracker(max_age=30), Visualizer(), stats,
                classes=COCO_CLASSES, width=int(prefs["width"]),
                show_fps=False, out_path=out_path, on_frame=on_frame,
            )
        except Exception as exc:
            error = f"Processing stopped: {exc}"
        finally:
            cap.release()
        bar.progress(1.0)
        if error:
            st.error(error)
        if stats.frames:
            persist_session(stats, uid, "Analysis", "Video file", started)
            snap = cv2.cvtColor(stats.snapshot, cv2.COLOR_BGR2RGB) if stats.snapshot is not None else None
            st.session_state["va_result"] = {
                "file": file_key, "unique": len(stats.unique), "fps": stats.frames / max(stats.elapsed, 1e-6),
                "elapsed": stats.elapsed, "snapshot": snap, "csv": stats.records_df().to_csv(index=False).encode("utf-8"),
                "video": out_path if stats.codec else None, "codec": stats.codec,
            }

    result = st.session_state.get("va_result")
    if result and result["file"] == file_key:
        spacer()
        cols = st.columns(3, gap="large")
        cols[0].markdown(kpi_html("Unique IDs", result["unique"]), unsafe_allow_html=True)
        cols[1].markdown(kpi_html("Average FPS", f"{result['fps']:.1f}"), unsafe_allow_html=True)
        cols[2].markdown(kpi_html("Time", f"{result['elapsed']:.0f} s"), unsafe_allow_html=True)
        spacer()
        video_ok = result["video"] and os.path.exists(result["video"])
        if video_ok and result["codec"] == "avc1":
            st.video(result["video"])
        elif result["snapshot"] is not None:
            show_image(st, result["snapshot"])
        d1, d2 = st.columns(2)
        d1.download_button("Download CSV", result["csv"], file_name="analysis_log.csv", mime="text/csv", key="dl_va_csv")
        if video_ok:
            with open(result["video"], "rb") as handle:
                d2.download_button("Download video", handle.read(), file_name="analysis.mp4", mime="video/mp4", key="dl_va_video")


def render_comparison() -> None:
    """Run two detector setups on the same footage with the same tracker."""
    keep_state(["cmp_ma", "cmp_ca", "cmp_mb", "cmp_cb", "cmp_frames"])
    st.session_state.setdefault("cmp_ma", "Fast")
    st.session_state.setdefault("cmp_mb", "Accurate")
    st.session_state.setdefault("cmp_ca", 0.40)
    st.session_state.setdefault("cmp_cb", 0.40)
    st.session_state.setdefault("cmp_frames", 200)

    uploaded = st.file_uploader("Video file", type=VIDEO_TYPES, key="cmp_file", label_visibility="collapsed")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        with keyed_container("card_cmp_a"):
            st.markdown('<div class="section-title">Setup A</div>', unsafe_allow_html=True)
            st.selectbox("Model", list(MODEL_FILES), key="cmp_ma")
            st.slider("Confidence", 0.10, 0.95, step=0.05, key="cmp_ca")
    with col_b:
        with keyed_container("card_cmp_b"):
            st.markdown('<div class="section-title">Setup B</div>', unsafe_allow_html=True)
            st.selectbox("Model", list(MODEL_FILES), key="cmp_mb")
            st.slider("Confidence", 0.10, 0.95, step=0.05, key="cmp_cb")
    st.slider("Frames to compare", 60, 600, step=20, key="cmp_frames")

    if uploaded is None:
        st.markdown('<div class="muted">Upload a video to compare two setups.</div>', unsafe_allow_html=True)
        return

    path = save_upload(uploaded)
    if st.button("Compare", type="primary", key="cmp_run"):
        runs: Dict[str, Dict[str, Any]] = {}
        bar = st.progress(0.0)
        setups = [("A", st.session_state["cmp_ma"], st.session_state["cmp_ca"]),
                  ("B", st.session_state["cmp_mb"], st.session_state["cmp_cb"])]
        limit = int(st.session_state["cmp_frames"])
        try:
            for index, (name, model, conf) in enumerate(setups):
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("This video file could not be opened.")
                stats = RunStats()

                def on_frame(frame: Any, valid: List[Tuple[Any, Any, Any]], fps: float,
                             _i: int = index, _s: RunStats = stats) -> None:
                    if _s.frames % 5 == 0:
                        bar.progress(min((_i + _s.frames / limit) / 2, 1.0))

                try:
                    run_pipeline(
                        cap, get_detector(MODEL_FILES[model], conf), ObjectTracker(max_age=30), Visualizer(), stats,
                        classes=COCO_CLASSES, width=640, show_fps=False, on_frame=on_frame, max_frames=limit,
                    )
                finally:
                    cap.release()
                snap = cv2.cvtColor(stats.snapshot, cv2.COLOR_BGR2RGB) if stats.snapshot is not None else None
                runs[name] = {
                    "model": model, "conf": conf, "fps": stats.avg_fps, "unique": len(stats.unique),
                    "detections": len(stats.records), "snapshot": snap,
                }
            bar.progress(1.0)
            st.session_state["cmp_result"] = {"file": f"{uploaded.name}:{uploaded.size}", "runs": runs}
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    result = st.session_state.get("cmp_result")
    if not result or result["file"] != f"{uploaded.name}:{uploaded.size}":
        return
    spacer()
    cols = st.columns(2, gap="large")
    for col, name in zip(cols, ("A", "B")):
        run = result["runs"].get(name)
        if not run:
            continue
        with col:
            st.markdown(
                mini_stats([("FPS", f"{run['fps']:.1f}"), ("Unique IDs", run["unique"]), ("Detections", run["detections"])]),
                unsafe_allow_html=True,
            )
            if run["snapshot"] is not None:
                show_image(st, run["snapshot"])
    spacer()
    tab_table, tab_chart = st.tabs(["Table", "Chart"])
    metrics = [("Average FPS", "fps"), ("Unique IDs", "unique"), ("Detections", "detections")]
    with tab_table:
        rows = [{"Metric": "Model", "Setup A": result["runs"]["A"]["model"], "Setup B": result["runs"]["B"]["model"]},
                {"Metric": "Confidence", "Setup A": result["runs"]["A"]["conf"], "Setup B": result["runs"]["B"]["conf"]}]
        for label, key in metrics:
            a, b = result["runs"]["A"][key], result["runs"]["B"][key]
            rows.append({"Metric": label, "Setup A": round(a, 1) if key == "fps" else a, "Setup B": round(b, 1) if key == "fps" else b})
        show_dataframe(pd.DataFrame(rows).astype(str))
    with tab_chart:
        chart_cols = st.columns(3)
        for col, (label, key) in zip(chart_cols, metrics):
            df = pd.DataFrame({"Setup": ["A", "B"], label: [result["runs"]["A"][key], result["runs"]["B"][key]]})
            fig = px.bar(df, x="Setup", y=label, color="Setup", color_discrete_map={"A": ACCENT, "B": GREY})
            fig.update_layout(showlegend=False)
            show_chart(style_fig(fig, 260), col)


# =============================================================================
# 10. PAGES: ANALYTICS AND REPORTS
# =============================================================================

def render_analytics() -> None:
    """Charts built from saved sessions."""
    keep_state(["an_range", "an_class"])
    clause, params = scope_clause("s")
    known = [r[0] for r in db_rows(
        f"SELECT DISTINCT t.class_name FROM session_tracks t JOIN sessions s ON s.id = t.session_id WHERE {clause} ORDER BY 1",
        params,
    )]
    today = dt.date.today()
    f1, f2 = st.columns([2, 1], gap="large")
    picked = f1.date_input("Date range", value=(today - dt.timedelta(days=30), today), key="an_range")
    chosen = f2.selectbox("Class", ["All"] + known, key="an_class")
    if isinstance(picked, (tuple, list)):
        start = picked[0]
        end = picked[1] if len(picked) > 1 else picked[0]
    else:
        start = end = picked
    window = (str(start), str(end))
    class_sql, class_params = ("", ()) if chosen == "All" else (" AND t.class_name = ?", (chosen,))
    point_sql, point_params = ("", ()) if chosen == "All" else (" AND p.class_name = ?", (chosen,))

    per_class = db_df(
        "SELECT t.class_name AS Class, COUNT(*) AS Objects FROM session_tracks t JOIN sessions s ON s.id = t.session_id "
        f"WHERE {clause} AND date(s.started_at) BETWEEN ? AND ?{class_sql} GROUP BY t.class_name ORDER BY Objects DESC",
        params + window + class_params,
    )
    over_time = db_df(
        "SELECT s.started_at AS Time, COUNT(t.id) AS Objects FROM sessions s LEFT JOIN session_tracks t ON t.session_id = s.id"
        f"{' AND t.class_name = ?' if chosen != 'All' else ''} "
        f"WHERE {clause} AND date(s.started_at) BETWEEN ? AND ? GROUP BY s.id ORDER BY s.started_at",
        class_params + params + window,
    )
    fps_df = db_df(
        f"SELECT s.started_at AS Time, s.avg_fps AS FPS FROM sessions s WHERE {clause} "
        "AND date(s.started_at) BETWEEN ? AND ? ORDER BY s.started_at",
        params + window,
    )
    points = db_df(
        "SELECT p.cx AS x, p.cy AS y FROM session_points p JOIN sessions s ON s.id = p.session_id "
        f"WHERE {clause} AND date(s.started_at) BETWEEN ? AND ?{point_sql} LIMIT 50000",
        params + window + point_params,
    )

    spacer()
    if over_time.empty:
        empty_state("No data in this range.")
        return

    top, bottom = st.columns(2, gap="large"), st.columns(2, gap="large")
    with top[0]:
        with keyed_container("card_an_class"):
            st.markdown('<div class="section-title">Objects per class</div>', unsafe_allow_html=True)
            fig = px.bar(per_class, x="Class", y="Objects")
            fig.update_traces(marker_color=ACCENT)
            show_chart(style_fig(fig))
    with top[1]:
        with keyed_container("card_an_time"):
            st.markdown('<div class="section-title">Objects over time</div>', unsafe_allow_html=True)
            fig = px.line(over_time, x="Time", y="Objects", markers=True)
            fig.update_traces(line_color=ACCENT, marker_color=ACCENT)
            show_chart(style_fig(fig))
    with bottom[0]:
        with keyed_container("card_an_fps"):
            st.markdown('<div class="section-title">Average FPS</div>', unsafe_allow_html=True)
            fig = px.line(fps_df, x="Time", y="FPS", markers=True)
            fig.update_traces(line_color=COPPER, marker_color=COPPER)
            show_chart(style_fig(fig))
    with bottom[1]:
        with keyed_container("card_an_heat"):
            st.markdown('<div class="section-title">Position heatmap</div>', unsafe_allow_html=True)
            if points.empty:
                st.markdown('<div class="muted">No position data.</div>', unsafe_allow_html=True)
            else:
                fig = go.Figure(go.Histogram2d(
                    x=points["x"], y=points["y"], nbinsx=24, nbinsy=14,
                    colorscale=[[0, "#1C1917"], [0.5, COPPER], [1, ACCENT]], showscale=False,
                ))
                fig.update_xaxes(range=[0, 1], showticklabels=False)
                fig.update_yaxes(range=[1, 0], showticklabels=False)
                show_chart(style_fig(fig))


REPORT_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Session report</title>
<style>
body{font-family:Inter,'Segoe UI',Arial,sans-serif;background:#141210;color:#F5F0EB;padding:40px;max-width:760px;margin:auto}
h1{font-size:28px;border-bottom:2px solid #D95A1A;padding-bottom:12px}
.row{display:flex;justify-content:space-between;background:#24201E;border:1px solid #373230;border-radius:10px;padding:14px 18px;margin:10px 0}
.row span:first-child{color:#A39A92} table{width:100%;border-collapse:collapse;margin-top:12px}
td,th{padding:10px;border-bottom:1px solid #373230;text-align:left} th{color:#A39A92;font-weight:600}
</style></head><body><h1>Session report</h1>__BODY__</body></html>"""


def build_report_html(session: pd.Series, classes: pd.DataFrame) -> str:
    """Return a standalone HTML report for one session."""
    fields = [("Session", session["name"]), ("Source", session["source"]), ("Started", session["started_at"]),
              ("Ended", session["ended_at"]), ("Frames", session["frames"]), ("Unique objects", session["objects"]),
              ("Average FPS", f"{float(session['avg_fps']):.1f}")]
    body = "".join(f'<div class="row"><span>{html.escape(k)}</span><span>{html.escape(str(v))}</span></div>' for k, v in fields)
    rows = "".join(f"<tr><td>{html.escape(str(r.Class))}</td><td>{int(r.Objects)}</td></tr>" for r in classes.itertuples())
    body += f"<table><tr><th>Class</th><th>Objects</th></tr>{rows}</table>"
    return REPORT_TEMPLATE.replace("__BODY__", body)


def render_reports() -> None:
    """Session list with per-session downloads."""
    clause, params = scope_clause("s")
    sessions = db_df(
        "SELECT s.id, s.name, s.source, s.started_at, s.ended_at, s.frames, s.objects, s.avg_fps, u.username AS user "
        f"FROM sessions s LEFT JOIN users u ON u.id = s.user_id WHERE {clause} ORDER BY s.id DESC",
        params,
    )
    if sessions.empty:
        empty_state("No sessions to report yet.")
        return
    table = sessions[["name", "source", "started_at", "objects", "avg_fps", "user"]].rename(columns={
        "name": "Session", "source": "Source", "started_at": "Started", "objects": "Objects", "avg_fps": "FPS", "user": "User"})
    table["FPS"] = table["FPS"].round(1)
    show_dataframe(table)

    spacer()
    names = dict(zip(sessions["id"], sessions["name"]))
    chosen = st.selectbox("Session", list(names), format_func=lambda i: names[i], key="rep_session")
    session = sessions[sessions["id"] == chosen].iloc[0]
    classes = db_df("SELECT class_name AS Class, COUNT(*) AS Objects FROM session_tracks WHERE session_id = ? GROUP BY class_name ORDER BY Objects DESC", (int(chosen),))
    tracks = db_df("SELECT track_id AS 'Track ID', class_name AS Class FROM session_tracks WHERE session_id = ?", (int(chosen),))

    cols = st.columns(3, gap="large")
    cols[0].markdown(kpi_html("Objects", int(session["objects"])), unsafe_allow_html=True)
    cols[1].markdown(kpi_html("Frames", int(session["frames"])), unsafe_allow_html=True)
    cols[2].markdown(kpi_html("Average FPS", f"{float(session['avg_fps']):.1f}"), unsafe_allow_html=True)
    spacer()
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV", tracks.to_csv(index=False).encode("utf-8"), file_name=f"session_{chosen}.csv", mime="text/csv", key="dl_rep_csv")
    d2.download_button("Download report", build_report_html(session, classes).encode("utf-8"), file_name=f"session_{chosen}.html", mime="text/html", key="dl_rep_html", type="primary")

    if st.session_state["auth_user"]["role"] == "Admin":
        spacer()
        confirm = st.checkbox("Confirm deletion of this session", key="rep_confirm")
        if st.button("Delete session", key="rep_delete") and confirm:
            for table_name, column in (("session_tracks", "session_id"), ("session_points", "session_id"), ("sessions", "id")):
                db_exec(f"DELETE FROM {table_name} WHERE {column} = ?", (int(chosen),))
            log_activity(st.session_state["auth_user"]["username"], "Delete session", str(chosen))
            st.rerun()


# =============================================================================
# 11. PAGES: SETTINGS, DOCUMENTATION, ADMIN
# =============================================================================

def render_settings() -> None:
    """Personal defaults and password change."""
    user = st.session_state["auth_user"]
    prefs = get_user_settings(user["id"])

    def push_to_live(values: Dict[str, Any]) -> None:
        st.session_state.update({
            "lm_conf": float(values["conf"]), "lm_model": values["model"],
            "lm_width": int(values["width"]), "lm_custom": list(values["classes"]),
        })

    with keyed_container("card_defaults"):
        st.markdown('<div class="section-title">Defaults</div>', unsafe_allow_html=True)
        with st.form("defaults_form"):
            c1, c2 = st.columns(2)
            model = c1.selectbox("Model", list(MODEL_FILES), index=list(MODEL_FILES).index(prefs["model"]))
            width = c2.slider("Frame width", 320, 1280, int(prefs["width"]), 80)
            conf = st.slider("Confidence", 0.10, 0.95, float(prefs["conf"]), 0.05)
            classes = st.multiselect("Classes", COCO_CLASSES, default=prefs["classes"])
            saved = st.form_submit_button("Save", type="primary")
        if saved:
            values = {"model": model, "conf": conf, "classes": classes or ["person"], "width": width}
            save_user_settings(user["id"], values)
            push_to_live(values)
            st.success("Saved.")
        if st.button("Reset to defaults", key="set_reset"):
            save_user_settings(user["id"], DEFAULT_PREFS)
            push_to_live(DEFAULT_PREFS)
            st.rerun()

    spacer()
    with keyed_container("card_password"):
        st.markdown('<div class="section-title">Password</div>', unsafe_allow_html=True)
        with st.form("password_form", clear_on_submit=True):
            current = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            again = st.text_input("Confirm new password", type="password")
            changed = st.form_submit_button("Update password", type="primary")
        if changed:
            if authenticate(user["username"], current) is None:
                st.error("Current password is incorrect.")
            elif len(new) < 8:
                st.error("Use at least 8 characters.")
            elif new != again:
                st.error("Passwords do not match.")
            else:
                set_password(user["id"], new)
                log_activity(user["username"], "Password changed")
                st.success("Password updated.")


def render_documentation() -> None:
    """Short help in four expanders."""
    with st.expander("Getting started", expanded=True):
        st.markdown(
            "- Open **Live Monitor**, choose a source and press **Start**.\n"
            "- Use **Video Analysis** to process a recorded file and download the results.\n"
            "- Open **Reports** to export any past session."
        )
    with st.expander("Concepts"):
        st.markdown(
            "- **Detection** finds objects in each frame.\n"
            "- **Tracking** keeps the same ID on an object across frames.\n"
            "- **NMS** removes duplicate boxes. **IoU** measures box overlap.\n"
            "- **Kalman filter** predicts where an object moves next.\n"
            "- **Hungarian algorithm** matches detections to existing tracks."
        )
    with st.expander("Troubleshooting"):
        st.markdown(
            "- **Camera not found:** try another index and close other apps using the camera.\n"
            "- **Low FPS:** pick the Fast model and reduce the frame width.\n"
            "- **No GPU:** check that your PyTorch build matches your CUDA driver.\n"
            "- **Stream will not open:** check the address, username, password and network."
        )
    with st.expander("FAQ"):
        st.markdown(
            "- **Network stream format?** `rtsp://user:password@192.168.1.10:554/stream1`, an HTTP stream, or a YouTube live link.\n"
            "- **Are passwords stored?** Stream passwords are used for the session only and are not saved.\n"
            "- **Where is data kept?** In the local `data` folder."
        )


def render_admin() -> None:
    """User management, activity, health and data tools."""
    me = st.session_state["auth_user"]
    if me["role"] != "Admin":
        st.error("You do not have access to this page.")
        return
    t_users, t_activity, t_health, t_data = st.tabs(["Users", "Activity", "System health", "Data"])

    with t_users:
        users = db_df("SELECT id, username, role, is_active AS active, created_at AS created FROM users ORDER BY id")
        show_dataframe(users)
        spacer()
        with keyed_container("card_user_new"):
            st.markdown('<div class="section-title">Create user</div>', unsafe_allow_html=True)
            with st.form("new_user_form", clear_on_submit=True):
                a, b, c = st.columns(3)
                name = a.text_input("Username")
                pwd = b.text_input("Password", type="password")
                role = c.selectbox("Role", ["User", "Admin"])
                created = st.form_submit_button("Create", type="primary")
            if created:
                if not name.strip() or len(pwd) < 8:
                    st.error("Enter a username and a password of at least 8 characters.")
                else:
                    pwd_hash, salt = hash_password(pwd)
                    try:
                        db_exec("INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                                (name.strip(), pwd_hash, salt, role))
                        log_activity(me["username"], "Create user", name.strip())
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("That username already exists.")
        spacer()
        with keyed_container("card_user_manage"):
            st.markdown('<div class="section-title">Manage user</div>', unsafe_allow_html=True)
            target = st.selectbox("User", users["username"].tolist(), key="adm_user")
            row = users[users["username"] == target].iloc[0]
            m1, m2 = st.columns(2)
            new_pwd = m1.text_input("New password", type="password", key="adm_pwd")
            new_role = m2.selectbox("Role", ["User", "Admin"], index=0 if row["role"] == "User" else 1, key="adm_role")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Reset password", key="adm_reset"):
                if len(new_pwd) < 8:
                    st.error("Use at least 8 characters.")
                else:
                    set_password(int(row["id"]), new_pwd)
                    log_activity(me["username"], "Reset password", target)
                    st.success("Password reset.")
            if b2.button("Apply role", key="adm_apply"):
                if target == me["username"]:
                    st.error("You cannot change your own role.")
                else:
                    db_exec("UPDATE users SET role = ? WHERE id = ?", (new_role, int(row["id"])))
                    st.rerun()
            if b3.button("Disable" if row["active"] else "Enable", key="adm_toggle"):
                if target == me["username"]:
                    st.error("You cannot disable your own account.")
                else:
                    db_exec("UPDATE users SET is_active = ? WHERE id = ?", (0 if row["active"] else 1, int(row["id"])))
                    st.rerun()
            confirm = st.checkbox("Confirm deletion", key="adm_confirm")
            if b4.button("Delete", key="adm_delete"):
                if target == me["username"]:
                    st.error("You cannot delete your own account.")
                elif not confirm:
                    st.error("Tick the confirmation box first.")
                else:
                    db_exec("DELETE FROM users WHERE id = ?", (int(row["id"]),))
                    db_exec("DELETE FROM user_settings WHERE user_id = ?", (int(row["id"]),))
                    log_activity(me["username"], "Delete user", target)
                    st.rerun()

    with t_activity:
        st.markdown('<div class="section-title">Sign-ins and actions</div>', unsafe_allow_html=True)
        show_dataframe(db_df("SELECT created_at AS Time, username AS User, action AS Action, detail AS Detail FROM activity_logs ORDER BY id DESC LIMIT 100"))
        spacer()
        st.markdown('<div class="section-title">Sessions</div>', unsafe_allow_html=True)
        show_dataframe(db_df(
            "SELECT s.started_at AS Started, u.username AS User, s.source AS Source, s.objects AS Objects, "
            "ROUND(s.avg_fps, 1) AS FPS FROM sessions s LEFT JOIN users u ON u.id = s.user_id ORDER BY s.id DESC LIMIT 100"))

    with t_health:
        cols = st.columns(4, gap="medium")
        cols[0].markdown(kpi_html("CPU", f"{psutil.cpu_percent(interval=0.2):.0f}%"), unsafe_allow_html=True)
        cols[1].markdown(kpi_html("Memory", f"{psutil.virtual_memory().percent:.0f}%"), unsafe_allow_html=True)
        cols[2].markdown(kpi_html("Disk", f"{psutil.disk_usage(os.path.abspath(os.sep)).percent:.0f}%"), unsafe_allow_html=True)
        cols[3].markdown(kpi_html("Device", device_label()), unsafe_allow_html=True)
        spacer()
        versions = [("OpenCV", cv2.__version__), ("Streamlit", st.__version__),
                    ("PyTorch", getattr(torch, "__version__", "not installed"))]
        try:
            import ultralytics
            versions.append(("Ultralytics", ultralytics.__version__))
        except Exception:
            versions.append(("Ultralytics", "not installed"))
        versions.append(("Model", "Loaded" if st.session_state.get("last_run") or st.session_state.get("is_tracking") else "Loads on first use"))
        show_dataframe(pd.DataFrame(versions, columns=["Component", "Status"]))

    with t_data:
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as handle:
                st.download_button("Export database", handle.read(), file_name="app.db", key="dl_db")
        spacer()
        with keyed_container("card_clear"):
            st.markdown('<div class="section-title">Clear data</div>', unsafe_allow_html=True)
            scope = st.selectbox("Data", ["Activity log", "Session history"], key="adm_clear_scope")
            sure = st.checkbox("I understand this cannot be undone", key="adm_clear_sure")
            if st.button("Clear", key="adm_clear"):
                if not sure:
                    st.error("Tick the confirmation box first.")
                else:
                    if scope == "Activity log":
                        db_exec("DELETE FROM activity_logs")
                    else:
                        for table_name in ("session_points", "session_tracks", "sessions"):
                            db_exec(f"DELETE FROM {table_name}")
                    st.success("Cleared.")


# =============================================================================
# 12. ROUTER
# =============================================================================

RENDERERS: Dict[str, Callable[[], None]] = {
    "Home": render_home,
    "Live Monitor": render_live_monitor,
    "Video Analysis": render_video_analysis,
    "Comparison Studio": render_comparison,
    "Analytics": render_analytics,
    "Reports": render_reports,
    "Settings": render_settings,
    "Documentation": render_documentation,
    "Admin Panel": render_admin,
}


def main() -> None:
    """Entry point: styles, sign-in gate, sidebar navigation and page routing."""
    ensure_streamlit_config()
    init_database()
    st.markdown(build_css(), unsafe_allow_html=True)

    if "auth_user" not in st.session_state:
        st.markdown(LOGIN_CSS, unsafe_allow_html=True)
        render_login()
        return

    user = st.session_state["auth_user"]
    available = [(n, i) for n, i in PAGES if n != "Admin Panel" or user["role"] == "Admin"]
    names = [n for n, _ in available]
    current = st.session_state.get("page", "Home")
    if current not in names:
        current = "Home"

    with st.sidebar:
        picked = option_menu(
            None, names, icons=[i for _, i in available], default_index=names.index(current),
            key=f"nav_{st.session_state.get('nav_nonce', 0)}", styles=NAV_STYLES,
        )
    page = picked if picked in names else current
    st.session_state["page"] = page
    if page != "Live Monitor":
        st.session_state["is_tracking"] = False

    render_top_bar(page)
    RENDERERS[page]()


if __name__ == "__main__":
    main()
