# """
# Speaker.py
# ACTIVE TTS:   ElevenLabs eleven_flash_v2_5
# INACTIVE TTS: Cartesia Sonic-3 — commented out, uncomment to switch

# NOISE MIXING: Commented out in _mix_noise usage in websocket_server.py
#   The _mix_noise function below is always available.
#   To re-enable: uncomment the noise block in websocket_server.py _process()
# """

# import os
# import base64
# import asyncio
# import httpx
# import io
# import hashlib

# os.environ["FFMPEG_BINARY"]  = r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
# os.environ["FFPROBE_BINARY"] = r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffprobe.exe"

# from pydub import AudioSegment

# # ── Noise mixing — available but disabled ────────────────────────────────────
# # Re-enable in websocket_server.py by uncommenting the noise block in _process()
# NOISE_FILE   = "freesound_community-office-ambience-24734 (1).mp3"
# NOISE_SLICES = 20


# def _mix_noise(voice_bytes: bytes, noise_slices: list, text: str) -> tuple[bytes, int]:
#     """
#     Mix voice with office ambience. Returns (bytes, duration_ms).
#     CURRENTLY UNUSED — disabled in websocket_server.py for speed.
#     Re-enable by uncommenting the noise block in _process().
#     """
#     try:
#         voice       = AudioSegment.from_file(io.BytesIO(voice_bytes)).fade_in(80)
#         duration_ms = len(voice)
#         hash_val    = int(hashlib.md5(text.encode()).hexdigest(), 16)
#         slice_idx   = hash_val % len(noise_slices)
#         noise_seg   = noise_slices[slice_idx]
#         loops       = (duration_ms // len(noise_seg)) + 2
#         noise       = (noise_seg * loops)[:duration_ms]
#         noise       = noise + 3
#         noise       = noise.low_pass_filter(4000)
#         combined    = voice.overlay(noise, gain_during_overlay=-3)
#         output      = io.BytesIO()
#         combined.export(output, format="mp3", bitrate="64k")
#         print("[Speaker] Ambience added")
#         return output.getvalue(), duration_ms
#     except Exception as e:
#         print(f"[Speaker] Noise failed: {e}")
#         return voice_bytes, get_duration_ms(voice_bytes)


# def get_duration_ms(audio_bytes: bytes) -> int:
#     try:
#         seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
#         return len(seg)
#     except Exception:
#         return int((len(audio_bytes) * 8) / (48 * 1000) * 1000)


# # ── ElevenLabs config (ACTIVE) ────────────────────────────────────────────────
# ELEVENLABS_URL      = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
# ELEVENLABS_MODEL    = "eleven_flash_v2_5"
# ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George

# # ── Cartesia config (INACTIVE — uncomment to switch) ─────────────────────────
# # CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"  # British Narration Lady
# # CARTESIA_MODEL    = "sonic-3"   # sonic-3: 90ms | sonic-turbo: 40ms

# # ── Recall.ai config ──────────────────────────────────────────────────────────
# RECALL_REGION   = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
# RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"


# class CartesiaSpeaker:
#     def __init__(self, bot_id: str = None):
#         self.elevenlabs_key = os.environ["ELEVENLABS_API_KEY"]
#         self.recall_key     = os.environ["RECALLAI_API_KEY"]
#         # self.cartesia_key = os.environ["CARTESIA_API_KEY"]  # uncomment for Cartesia
#         self.bot_id         = bot_id

#         # Noise slices — pre-loaded, not used until re-enabled
#         base_dir   = os.path.dirname(os.path.abspath(__file__))
#         noise_path = os.path.join(base_dir, NOISE_FILE)
#         self._noise_slices = []
#         try:
#             full_noise = AudioSegment.from_file(noise_path)
#             slice_len  = len(full_noise) // NOISE_SLICES
#             self._noise_slices = [
#                 full_noise[i * slice_len:(i + 1) * slice_len]
#                 for i in range(NOISE_SLICES)
#             ]
#             print(f"[Speaker] Noise pre-sliced into {NOISE_SLICES} chunks ({slice_len}ms each)")
#         except Exception as e:
#             print(f"[Speaker] Noise load failed (not critical): {e}")

#         self._base_noise = self._noise_slices if self._noise_slices else None

#         limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

#         # ── ACTIVE: ElevenLabs client ─────────────────────────────────────────
#         self._elevenlabs_client  = httpx.AsyncClient(timeout=30, limits=limits, http2=True)
#         self._elevenlabs_headers = {
#             "xi-api-key":   self.elevenlabs_key,
#             "Content-Type": "application/json",
#         }

#         # ── INACTIVE: Cartesia client ─────────────────────────────────────────
#         # To switch to Cartesia:
#         #   1. Uncomment these lines
#         #   2. Uncomment cartesia_key above
#         #   3. Comment out _elevenlabs_client and _elevenlabs_headers above
#         #   4. Swap _synthesise methods below
#         #
#         # self._cartesia_client  = httpx.AsyncClient(timeout=30, limits=limits, http2=True)
#         # self._cartesia_headers = {
#         #     "Authorization":    f"Bearer {self.cartesia_key}",
#         #     "Cartesia-Version": "2025-04-16",
#         #     "Content-Type":     "application/json",
#         # }

#         # Recall inject client
#         self._recall_client  = httpx.AsyncClient(timeout=30, limits=limits, http2=True)
#         self._recall_headers = {
#             "Authorization": f"Token {self.recall_key}",
#             "Content-Type":  "application/json",
#             "accept":        "application/json",
#         }

#     # ── ACTIVE: ElevenLabs TTS ────────────────────────────────────────────────
#     async def _synthesise(self, text: str) -> bytes:
#         """ElevenLabs eleven_flash_v2_5 — ultra-realistic voice."""
#         payload = {
#             "text":     text,
#             "model_id": ELEVENLABS_MODEL,
#             "voice_settings": {
#                 "stability":         0.35,
#                 "similarity_boost":  0.75,
#                 "style":             0.0,
#                 "use_speaker_boost": True,
#             },
#             "output_format": "mp3_44100_64",
#         }
#         response = await self._elevenlabs_client.post(
#             ELEVENLABS_URL.format(voice_id=ELEVENLABS_VOICE_ID),
#             headers=self._elevenlabs_headers,
#             json=payload,
#         )
#         response.raise_for_status()
#         return response.content

#     # ── INACTIVE: Cartesia TTS ────────────────────────────────────────────────
#     # To switch to Cartesia:
#     #   1. Comment out the ElevenLabs _synthesise above
#     #   2. Uncomment this method
#     #
#     # async def _synthesise(self, text: str) -> bytes:
#     #     """Cartesia Sonic-3 — 90ms first byte."""
#     #     response = await self._cartesia_client.post(
#     #         "https://api.cartesia.ai/tts/bytes",
#     #         headers=self._cartesia_headers,
#     #         json={
#     #             "model_id":   CARTESIA_MODEL,
#     #             "transcript": text,
#     #             "voice":      {"mode": "id", "id": CARTESIA_VOICE_ID},
#     #             "language":   "en",
#     #             "output_format": {
#     #                 "container":   "mp3",
#     #                 "sample_rate": 44100,
#     #                 "bit_rate":    128000,
#     #             },
#     #         },
#     #     )
#     #     response.raise_for_status()
#     #     return response.content

#     async def _inject_into_meeting(self, b64_audio: str):
#         if not self.bot_id:
#             print("[Speaker] No bot_id — skipping inject")
#             return
#         payload  = {"kind": "mp3", "b64_data": b64_audio}
#         response = await self._recall_client.post(
#             f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
#             headers=self._recall_headers,
#             json=payload,
#         )
#         if response.status_code not in (200, 201):
#             print(f"[Speaker] Inject error {response.status_code}: {response.text}")
#         else:
#             print("[Speaker] Audio injected")

#     async def close(self):
#         await asyncio.gather(
#             self._elevenlabs_client.aclose(),
#             self._recall_client.aclose(),
#             # self._cartesia_client.aclose(),  # uncomment if Cartesia re-enabled
#         )

"""
Speaker.py
ACTIVE TTS:   ElevenLabs eleven_flash_v2_5
INACTIVE TTS: Cartesia Sonic-turbo — commented out

NOISE MIXING: Disabled — re-enable in websocket_server.py _process()
"""

import os
import base64
import asyncio
import httpx
import io
import hashlib
import platform
if platform.system() == "Windows":
    os.environ["FFMPEG_BINARY"]  = r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    os.environ["FFPROBE_BINARY"] = r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffprobe.exe"

from pydub import AudioSegment

NOISE_FILE   = "freesound_community-office-ambience-24734 (1).mp3"
NOISE_SLICES = 20


def _mix_noise(voice_bytes: bytes, noise_slices: list, text: str) -> tuple[bytes, int]:
    try:
        voice       = AudioSegment.from_file(io.BytesIO(voice_bytes)).fade_in(80)
        duration_ms = len(voice)
        hash_val    = int(hashlib.md5(text.encode()).hexdigest(), 16)
        slice_idx   = hash_val % len(noise_slices)
        noise_seg   = noise_slices[slice_idx]
        loops       = (duration_ms // len(noise_seg)) + 2
        noise       = (noise_seg * loops)[:duration_ms]
        noise       = noise + 3
        noise       = noise.low_pass_filter(4000)
        combined    = voice.overlay(noise, gain_during_overlay=-3)
        output      = io.BytesIO()
        combined.export(output, format="mp3", bitrate="64k")
        return output.getvalue(), duration_ms
    except Exception as e:
        print(f"[Speaker] Noise failed: {e}")
        return voice_bytes, get_duration_ms(voice_bytes)


def get_duration_ms(audio_bytes: bytes) -> int:
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return len(seg)
    except Exception:
        return int((len(audio_bytes) * 8) / (48 * 1000) * 1000)


# ── Cartesia config (INACTIVE) ────────────────────────────────────────────────
# CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
# CARTESIA_MODEL    = "sonic-turbo"

# ── Recall.ai config ──────────────────────────────────────────────────────────
RECALL_REGION   = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"


class CartesiaSpeaker:
    def __init__(self, bot_id: str = None):
        import Speaker as _self_module
        print(f"[Speaker] Loaded from: {_self_module.__file__}")
        self.recall_key     = os.environ["RECALLAI_API_KEY"]
        self.bot_id         = bot_id

        base_dir   = os.path.dirname(os.path.abspath(__file__))
        noise_path = os.path.join(base_dir, NOISE_FILE)
        self._noise_slices = []
        try:
            full_noise = AudioSegment.from_file(noise_path)
            slice_len  = len(full_noise) // NOISE_SLICES
            self._noise_slices = [
                full_noise[i * slice_len:(i + 1) * slice_len]
                for i in range(NOISE_SLICES)
            ]
        except Exception as e:
            print(f"[Speaker] Noise load failed (not critical): {e}")

        self._base_noise = self._noise_slices if self._noise_slices else None
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        # ── Multi-key Cartesia setup ─────────────────────────────────────
        self._cartesia_keys = []
        for key_name in ["CARTESIA_API_KEY", "CARTESIA_API_KEY_2", "CARTESIA_API_KEY_3", "CARTESIA_API_KEY_4", "CARTESIA_API_KEY_5"]:
            val = os.environ.get(key_name, "").strip()
            if val:
                self._cartesia_keys.append(val)

        if not self._cartesia_keys:
            raise ValueError("No CARTESIA_API_KEY found in environment")

        print(f"[Speaker] {len(self._cartesia_keys)} Cartesia key(s) loaded")
        self._key_index = 0

        self._cartesia_client = httpx.AsyncClient(timeout=30, limits=limits)

        self._recall_client  = httpx.AsyncClient(timeout=30, limits=limits)
        self._recall_headers = {
            "Authorization": f"Token {self.recall_key}",
            "Content-Type":  "application/json",
            "accept":        "application/json",
        }

    async def warmup(self):
        """Validate all keys + warm up TCP connection."""
        valid_keys = []
        for i, key in enumerate(self._cartesia_keys):
            try:
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Cartesia-Version": "2025-04-16",
                    "Content-Type": "application/json",
                }
                response = await self._cartesia_client.post(
                    "https://api.cartesia.ai/tts/bytes",
                    headers=headers,
                    json={
                        "model_id": "sonic-turbo",
                        "transcript": "hi",
                        "voice": {"mode": "id", "id": "79a125e8-cd45-4c13-8a67-188112f4dd22"},
                        "language": "en",
                        "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 192000},
                    },
                )
                if response.status_code in (200, 201):
                    valid_keys.append(key)
                    print(f"[Speaker] ✅ Cartesia key #{i+1} valid")
                else:
                    print(f"[Speaker] ❌ Cartesia key #{i+1} invalid ({response.status_code})")
            except Exception as e:
                print(f"[Speaker] ❌ Cartesia key #{i+1} failed: {e}")

        if valid_keys:
            self._cartesia_keys = valid_keys
            print(f"[Speaker] ✅ {len(valid_keys)} valid key(s), Cartesia warmed up")
        else:
            print(f"[Speaker] ⚠️  No valid Cartesia keys!")

    def _next_cartesia_headers(self) -> dict:
        key = self._cartesia_keys[self._key_index % len(self._cartesia_keys)]
        self._key_index += 1
        return {
            "Authorization":    f"Bearer {key}",
            "Cartesia-Version": "2025-04-16",
            "Content-Type":     "application/json",
        }

    async def _synthesise(self, text: str) -> bytes:
        headers = self._next_cartesia_headers()
        key_num = (self._key_index - 1) % len(self._cartesia_keys) + 1
        print(f"[Speaker] TTS via Cartesia (key #{key_num})...")
        response = await self._cartesia_client.post(
            "https://api.cartesia.ai/tts/bytes",
            headers=headers,
            json={
                "model_id":   "sonic-turbo",
                "transcript": text,
                "voice":      {"mode": "id", "id": "79a125e8-cd45-4c13-8a67-188112f4dd22"},
                "language":   "en",
                "output_format": {
                    "container":   "mp3",
                    "sample_rate": 44100,
                    "bit_rate":    192000,
                },
            },
        )
        response.raise_for_status()
        return response.content

    async def _inject_into_meeting(self, b64_audio: str):
        if not self.bot_id:
            print("[Speaker] No bot_id — skipping inject")
            return

        # Debug: save MP3 that gets sent to Recall.ai
        if os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes"):
            try:
                import base64 as _b64
                raw = _b64.b64decode(b64_audio)
                self._debug_audio_counter = getattr(self, '_debug_audio_counter', 0) + 1
                fname = f"debug_inject_{self._debug_audio_counter:03d}.mp3"
                with open(fname, "wb") as f:
                    f.write(raw)
                print(f"[Speaker] 🔧 Debug saved: {fname} ({len(raw)} bytes)")
            except Exception as e:
                print(f"[Speaker] Debug save failed: {e}")

        response = await self._recall_client.post(
            f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
            headers=self._recall_headers,
            json={"kind": "mp3", "b64_data": b64_audio},
        )
        if response.status_code not in (200, 201):
            print(f"[Speaker] Inject error {response.status_code}: {response.text}")
        else:
            print("[Speaker] Audio injected")

    async def stop_audio(self):
        if not self.bot_id:
            return
        try:
            response = await self._recall_client.delete(
                f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
                headers=self._recall_headers,
            )
            if response.status_code == 204:
                print("[Speaker] ⏹️  Audio stopped")
            else:
                print(f"[Speaker] Stop audio: {response.status_code}")
        except Exception as e:
            print(f"[Speaker] Stop audio error: {e}")

    async def close(self):
        await asyncio.gather(
            self._cartesia_client.aclose(),
            self._recall_client.aclose(),
        )