"""
server.py — Railway/Local entry point with JWT authentication

Endpoints:
  POST /auth/login    → returns JWT token
  POST /start         → protected: Sam joins a meeting
  POST /stop          → protected: Sam leaves the meeting
  GET  /health        → public health check
  GET  /status        → protected: current bot status
  GET  /              → serves frontend (index.html)
  WS   /ws            → Recall.ai WebSocket (unchanged)
"""

import asyncio
import os
import time
import json
import hashlib
import hmac
import base64
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

from websocket_server import WebSocketServer
from recall_bot import RecallBot

PORT = int(os.environ.get("PORT", 8000))

# ── JWT Config ────────────────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-please")
JWT_EXPIRY = 24 * 3600  # 24 hours

# ── User store (env-based) ────────────────────────────────────────────────────
USERS = {}
admin_user = os.environ.get("ADMIN_USERNAME", "admin")
admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
USERS[admin_user] = admin_pass

for i in range(1, 11):
    name = os.environ.get(f"USER_{i}_NAME", "").strip()
    pwd = os.environ.get(f"USER_{i}_PASS", "").strip()
    if name and pwd:
        USERS[name] = pwd

print(f"[Auth] {len(USERS)} user(s) configured")


# ── Minimal JWT (no PyJWT needed) ─────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)

def jwt_encode(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header).encode())
    p = _b64url_encode(json.dumps(payload).encode())
    sig = hmac.new(JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"

def jwt_decode(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, p, s = parts
        expected = hmac.new(JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(s)):
            return None
        payload = json.loads(_b64url_decode(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

def _get_user(request) -> dict | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return jwt_decode(auth[7:])


# ── State ─────────────────────────────────────────────────────────────────────
active_bots = {}  # username -> {"bot", "bot_id", "meeting_url", "started_at"}
active_server = None
_start_time = time.time()


# ── Auth ──────────────────────────────────────────────────────────────────────

async def handle_login(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if username not in USERS or USERS[username] != password:
        return web.json_response({"error": "Invalid credentials"}, status=401)

    token = jwt_encode({
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY,
    })
    return web.json_response({"token": token, "username": username, "expires_in": JWT_EXPIRY})


# ── Bot control ───────────────────────────────────────────────────────────────

async def handle_start(request: web.Request) -> web.Response:
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    meeting_url = data.get("meeting_url", "").strip()
    if not meeting_url:
        return web.json_response({"error": "meeting_url required"}, status=400)

    username = user["sub"]

    # Stop existing bot if any
    if username in active_bots:
        try:
            await active_bots[username]["bot"].leave()
        except Exception:
            pass
        del active_bots[username]

    # WebSocket URL — Railway uses RAILWAY_PUBLIC_DOMAIN, local uses TUNNEL_URL
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    tunnel = os.environ.get("TUNNEL_URL", "").strip().rstrip("/")
    if domain:
        ws_url = f"wss://{domain}/ws"
    elif tunnel:
        ws_url = tunnel.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/ws"):
            ws_url = ws_url + "/ws"
    else:
        return web.json_response({
            "error": "No public URL configured. Set TUNNEL_URL in .env for local dev, or deploy to Railway."
        }, status=400)

    print(f"[Server] {username} → deploying Sam to {meeting_url}")

    try:
        bot = RecallBot()
        bot_id = await bot.join(meeting_url, ws_url)
        if active_server:
            active_server.speaker.bot_id = bot_id

        active_bots[username] = {
            "bot": bot, "bot_id": bot_id,
            "meeting_url": meeting_url, "started_at": time.time(),
        }
        return web.json_response({"status": "joined", "bot_id": bot_id, "meeting_url": meeting_url})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_stop(request: web.Request) -> web.Response:
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)

    username = user["sub"]
    if username not in active_bots:
        return web.json_response({"status": "no active bot"})

    try:
        await active_bots[username]["bot"].leave()
    except Exception:
        pass
    del active_bots[username]
    return web.json_response({"status": "left"})


async def handle_status(request: web.Request) -> web.Response:
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)

    info = active_bots.get(user["sub"])
    if info:
        return web.json_response({
            "active": True, "bot_id": info["bot_id"],
            "meeting_url": info["meeting_url"],
            "uptime_seconds": int(time.time() - info["started_at"]),
        })
    return web.json_response({"active": False})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok", "active_bots": len(active_bots),
        "uptime": int(time.time() - _start_time),
    })


# ── Frontend ──────────────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.Response:
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        return web.FileResponse(html_path)
    return web.Response(text="index.html not found", status=404)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global active_server

    server = WebSocketServer(port=PORT, bot_id=None)
    active_server = server

    server.app.router.add_post("/auth/login", handle_login)
    server.app.router.add_post("/start", handle_start)
    server.app.router.add_post("/stop", handle_stop)
    server.app.router.add_get("/status", handle_status)
    server.app.router.add_get("/api/health", handle_health)
    server.app.router.add_get("/", handle_index)

    await server.start()

    print(f"[Server] Running on port {PORT}")
    print(f"[Server] Frontend: http://localhost:{PORT}/")
    print(f"[Server] Credentials: {admin_user} / {'*' * len(admin_pass)}")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

if __name__ == "__main__":
    asyncio.run(main())