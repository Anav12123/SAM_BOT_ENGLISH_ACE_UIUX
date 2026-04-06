"""
websocket_server.py — Production voice pipeline (v2: EOT-first architecture)

Pipeline:
  Deepgram final transcript → Buffer → EOT check (~150ms) →
    COMPLETE → 500ms straggler wait → process
    INCOMPLETE → 4s safety timeout → process anyway

No more VAD flush, no debounce timers, no peak_rms filtering.
VAD is kept only for audio monitoring and interrupt detection.
"""

import asyncio
import json
import time
import base64
import os
import re as _re
import random
from aiohttp import web
import aiohttp
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent, FILLERS
from Speaker import CartesiaSpeaker, _mix_noise
from vad import RmsVAD


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"

WORDS_PER_SECOND = 3.2

# ── Ack phrases — ignored during search ──────────────────────────────────────
_ACK_PHRASES = frozenset({
    "sure", "ok", "okay", "yeah", "yes", "go ahead", "alright",
    "right", "hmm", "mhm", "cool", "got it", "fine", "yep", "yup",
    "carry on", "go on", "continue", "waiting", "i'm waiting",
    "i am waiting", "no problem", "take your time", "np",
    "hello", "hi", "hey", "huh", "what", "sorry",
})

# ── Interrupt ack phrases — pre-baked at startup for instant playback ─────────
_INTERRUPT_ACKS = [
    "Oh sorry, go ahead.",
    "My bad, what were you saying?",
    "Sure, I'm listening.",
    "Oh, go on.",
]

# ── Fix Deepgram misrecognitions ─────────────────────────────────────────────
_TRANSCRIPTION_FIXES = [
    (_re.compile(r'\b(?:NF\s*Cloud|Enuf\s*Cloud|Enough\s*Cloud|Nav\s*Cloud|Anav\s*Cloud|Arnav\s*Cloud|Anab\s*Cloud|NFClouds?|EnoughClouds?|NavClouds?|AnavCloud)\b', _re.IGNORECASE), 'AnavClouds'),
    (_re.compile(r'\b(?:Sales\s*Force|Sells\s*Force|Cells\s*Force|SalesForce)\b', _re.IGNORECASE), 'Salesforce'),
]

def _fix_transcription(text: str) -> str:
    result = text
    for pattern, replacement in _TRANSCRIPTION_FIXES:
        result = pattern.sub(replacement, result)
    if result != text:
        print(f"[Transcript Fix] \"{text}\" → \"{result}\"")
    return result

def _is_ack(text: str) -> bool:
    """Check if text is purely acknowledgement phrases."""
    fragments = _re.split(r'[.!?,]+', text.strip().lower())
    return all(f.strip() in _ACK_PHRASES or f.strip() == "" for f in fragments) and text.strip() != ""


class WebSocketServer:
    # ── Timing constants ──────────────────────────────────────────────────────
    STRAGGLER_WAIT  = 0.8   # seconds after RESPOND before processing (catch split transcripts)
    WAIT_TIMEOUT    = 2.0   # seconds after WAIT — force process (speaker went silent)

    def __init__(self, port: int = 8000, bot_id: str = None):
        self.port           = port
        self.trigger        = TriggerDetector()
        self.agent          = PMAgent()
        self.speaker        = CartesiaSpeaker(bot_id=bot_id)
        self._speaking      = False
        self._audio_playing = False
        self._convo_history = deque(maxlen=10)

        self._current_task:    asyncio.Task | None = None
        self._current_text:    str  = ""
        self._current_speaker: str  = ""
        self._interrupt_event: asyncio.Event = asyncio.Event()
        self._generation:      int  = 0
        self._last_spoke_at:   float = 0.0

        self._buffer:      list = []     # [(speaker, text, timestamp)]
        self._partial_text:    str = ""
        self._partial_speaker: str = ""
        self._last_flushed_text: str = ""

        # Interrupt handling — pre-baked ack audio for instant playback
        self._was_interrupted: bool = False
        self._interrupt_ack_audio: list[tuple[str, bytes]] = []

        # EOT check task — the ONLY path to processing
        self._eot_task: asyncio.Task | None = None

        # Search state
        self._searching = False
        self._pending_searches: list[tuple[str, asyncio.Task]] = []

        # TTS rate limiter — matches number of Cartesia keys
        self._tts_semaphore = asyncio.Semaphore(4)

        # VAD — kept for audio monitoring and interrupt detection only
        self._vad = RmsVAD()
        # Debug: save raw audio to file for analysis
        self.DEBUG_SAVE_AUDIO = os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes")
        self._debug_audio_file = None

        self.app = web.Application()
        self.app.router.add_get("/ws",     self.handle_websocket)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "speaking": self._speaking, "searching": self._searching})

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        print(f"[{ts()}] ✅ Recall.ai WebSocket connected")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._handle_event(msg.data)
                    except Exception as e:
                        print(f"[{ts()}] ⚠️  Event handler error: {e}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[{ts()}] ⚠️  WS error: {ws.exception()}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
        except Exception as e:
            print(f"[{ts()}] WS handler error: {e}")
        finally:
            print(f"[{ts()}] WebSocket disconnected")
        return ws

    # ══════════════════════════════════════════════════════════════════════════
    # Event dispatch
    # ══════════════════════════════════════════════════════════════════════════

    async def _handle_event(self, raw: str):
        t = time.time()
        try:
            payload = json.loads(raw)
        except Exception:
            return

        event = payload.get("event", "")

        # ── Transcript (final) ────────────────────────────────────────
        if event == "transcript.data":
            inner   = payload.get("data", {}).get("data", {})
            words   = inner.get("words", [])
            text    = " ".join(w.get("text", "") for w in words).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if not text or speaker.lower() == "sam":
                return

            text = _fix_transcription(text)
            # Clean Gladia quirks: double spaces, leading dashes
            text = _re.sub(r'\s+', ' ', text).strip().lstrip("-–— ").strip()
            if not text:
                return
            buf_words = sum(len(txt.split()) for _, txt, _ in self._buffer)
            print(f"\n[{ts()}] [{speaker}] {text}  ⏱ {elapsed(t)}  🔧 speaking={self._speaking} audio={self._audio_playing} buf={buf_words}w")

            # ── Post-interrupt: discard this fragment, play ack ──
            if self._was_interrupted:
                self._was_interrupted = False
                self._buffer.clear()
                self._partial_text = ""
                self._partial_speaker = ""
                self._vad.end_turn()
                self._last_flushed_text = ""
                print(f"[{ts()}] 🙏 Post-interrupt — discarding \"{text}\", playing ack")
                self.agent.log_exchange(speaker, text)
                await self._play_interrupt_ack()
                return

            # Final replaces any partial for this speaker
            self._partial_text = ""
            self._partial_speaker = ""

            # Skip if this text was already flushed
            if self._last_flushed_text:
                flushed = self._last_flushed_text.lower().strip()
                incoming = text.lower().strip()
                flushed_words = set(flushed.split())
                incoming_words = set(incoming.split())
                overlap = len(flushed_words & incoming_words)
                max_len = max(len(flushed_words), len(incoming_words), 1)
                similarity = overlap / max_len
                if incoming in flushed or flushed in incoming or similarity >= 0.7:
                    print(f"[{ts()}] 🔕 Skipping — already flushed (sim={similarity:.0%}): \"{text}\"")
                    self._last_flushed_text = ""
                    self.agent.log_exchange(speaker, text)
                    return
            self._last_flushed_text = ""

            # Store in RAG immediately
            self.agent.log_exchange(speaker, text)

            # ── Ack during processing/search → ignore ─────────
            # Check BEFORE interrupt logic — acks like "Sure", "Ok" shouldn't cancel anything
            if self._speaking and self._current_speaker == speaker:
                # Clean transcript artifacts (leading dashes, etc) for ack check
                clean_for_ack = text.lstrip("-–— ").strip()
                if _is_ack(clean_for_ack):
                    print(f"[{ts()}] 🔕 Ack during processing — ignored: \"{text}\"")
                    return

            # ── Different speaker while Sam is processing/speaking ──────
            if self._speaking and self._current_speaker != speaker:
                if self._audio_playing:
                    print(f"[{ts()}] ⚡ INTERRUPT — {speaker} cut in (audio playing)")
                    await self.speaker.stop_audio()
                    self._interrupt_event.set()
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    await asyncio.sleep(0.1)
                    self._speaking = False
                    self._audio_playing = False
                    self._searching = False
                    self._buffer.clear()
                    self._partial_text = ""
                    self._partial_speaker = ""
                    self._vad.end_turn()
                    await self._play_interrupt_ack()
                    return
                else:
                    # Pre-audio: cancel processing, re-buffer, re-check EOT
                    print(f"[{ts()}] 📥 {speaker} added text (pre-audio) — cancelling + re-buffering")
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    self._interrupt_event.set()
                    await asyncio.sleep(0.05)
                    self._speaking = False
                    self._searching = False
                    self._buffer.append((speaker, text, t))
                    self._schedule_eot_check(speaker)
                    return

            # ── Same speaker adds more while Sam is processing/speaking ──
            if self._speaking and self._current_speaker == speaker:
                if self._audio_playing:
                    print(f"[{ts()}] ⚡ INTERRUPT — {speaker} cut in (same speaker, audio playing)")
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    await self.speaker.stop_audio()
                    self._interrupt_event.set()
                    await asyncio.sleep(0.1)
                    self._speaking = False
                    self._audio_playing = False
                    self._searching = False
                    self._buffer.clear()
                    self._partial_text = ""
                    self._partial_speaker = ""
                    self._vad.end_turn()
                    await self._play_interrupt_ack()
                    return
                else:
                    # Pre-audio: bot is preparing response but hasn't spoken yet
                    # Short utterances (≤6 words) from same speaker are likely
                    # commentary ("you know?", "very confused"), NOT a new question
                    word_count = len(text.split())
                    if word_count <= 8:
                        print(f"[{ts()}] 💬 Commentary during processing — absorbed: \"{text}\" ({word_count}w)")
                        self.agent.log_exchange(speaker, text)  # still log to RAG
                        return
                    # Longer text = user is actually asking something new, cancel + re-buffer
                    buf_words = sum(len(txt.split()) for _, txt, _ in self._buffer)
                    print(f"[{ts()}] 📥 New question (pre-audio) — cancelling + re-buffering ({buf_words}+{word_count} words)")
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    self._interrupt_event.set()
                    await asyncio.sleep(0.05)
                    self._speaking = False
                    self._searching = False
                    self._buffer.append((speaker, text, t))
                    self._schedule_eot_check(speaker)
                    return

            # ── Normal: buffer + run EOT check ────────────────────
            self._buffer.append((speaker, text, t))
            buf_total = sum(len(txt.split()) for _, txt, _ in self._buffer)
            print(f"[{ts()}] 📦 Buffered: {buf_total} words total → scheduling EOT check")
            self._schedule_eot_check(speaker)

        # ── Transcript (partial — updates as user speaks) ──
        elif event == "transcript.partial_data":
            inner   = payload.get("data", {}).get("data", {})
            words   = inner.get("words", [])
            text    = " ".join(w.get("text", "") for w in words).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if not text or speaker.lower() == "sam":
                return

            text = _fix_transcription(text)
            text = _re.sub(r'\s+', ' ', text).strip().lstrip("-–— ").strip()
            if not text:
                return
            self._partial_text = text
            self._partial_speaker = speaker
            # Partial means user is still speaking — cancel any pending EOT check
            if self._eot_task and not self._eot_task.done():
                self._eot_task.cancel()

        # ── Speech OFF ──────────────────────────────────────────────
        elif event == "participant_events.speech_off":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            print(f"[{ts()}] 🔇 {speaker} stopped speaking")

        elif event == "participant_events.speech_on":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            print(f"[{ts()}] 🎤 {speaker} started speaking")

            # Interrupt detection: only when bot is audibly playing
            if self._speaking and self._current_speaker != speaker:
                if self._audio_playing:
                    print(f"[{ts()}] ⚡ INTERRUPT (speech_on) — {speaker} cut in")
                    await self.speaker.stop_audio()
                    await asyncio.sleep(0.1)
                    self._interrupt_event.set()
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    self._speaking = False
                    self._audio_playing = False
                    self._was_interrupted = True

        # ── Raw audio → VAD monitoring only (no flush decisions) ─────
        elif event == "audio_mixed_raw.data":
            if not self._vad.ready or self._audio_playing:
                return

            audio_b64 = payload.get("data", {}).get("data", {}).get("buffer", "")
            if not audio_b64:
                return

            try:
                pcm_bytes = base64.b64decode(audio_b64)

                if self.DEBUG_SAVE_AUDIO:
                    if self._debug_audio_file is None:
                        self._debug_audio_file = open("debug_audio.raw", "wb")
                        print(f"[{ts()}] 🔧 Debug audio saving to debug_audio.raw")
                    self._debug_audio_file.write(pcm_bytes)

                rms_values = self._vad.process_chunk(pcm_bytes)
                for rms in rms_values:
                    self._vad.update_state(rms)

                # Periodic debug logging
                if not hasattr(self, '_audio_event_count'):
                    self._audio_event_count = 0
                    self._max_conf = 0.0
                self._audio_event_count += 1
                if rms_values:
                    self._max_conf = max(self._max_conf, max(rms_values))

                if self._audio_event_count == 1:
                    print(f"[{ts()}] 🔊 First audio_mixed_raw received ({len(pcm_bytes)} bytes)")
                elif self._audio_event_count % 200 == 0:
                    print(f"[{ts()}] 🔊 Audio#{self._audio_event_count} rms={self._vad.last_confidence:.4f} max={self._max_conf:.4f} buf={len(self._buffer)}")

            except Exception as e:
                print(f"[{ts()}] ⚠️  VAD error: {e}")

        elif event == "participant_events.join":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] 👋 {name} joined")
                asyncio.create_task(self._greet(name, t))

        elif event == "participant_events.leave":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] 👋 {name} left")

    # ══════════════════════════════════════════════════════════════════════════
    # EOT-first turn detection (the ONLY path to processing)
    # ══════════════════════════════════════════════════════════════════════════

    def _schedule_eot_check(self, speaker: str):
        """Cancel any pending EOT check and schedule a new one."""
        if self._eot_task and not self._eot_task.done():
            self._eot_task.cancel()
        self._eot_task = asyncio.create_task(self._run_eot_check(speaker))

    async def _run_eot_check(self, speaker: str):
        """Context-aware turn detection: should Sam respond now or wait?
        
        Flow:
          1. Run EOT classifier with conversation context (~150ms)
          2. RESPOND → wait 0.8s for stragglers → process
          3. WAIT → wait 6s (speaker still going) → force process if no new text
          
        Any new transcript (final or partial) cancels this task and triggers a fresh check.
        """
        try:
            result = self._get_buffer_text()
            if not result or self._speaking:
                return

            spk, full_text, t0 = result
            word_count = len(full_text.split())

            # Get conversation context for the EOT classifier
            context = "\n".join(self._convo_history)
            ctx_count = len(self._convo_history)
            print(f"[{ts()}] 📋 EOT context ({ctx_count} entries): {list(self._convo_history)[-3:] if self._convo_history else '(EMPTY)'}")

            # Ask the LLM: should Sam respond now or wait?
            decision = await self.agent.check_end_of_turn(full_text, context)

            if decision == "RESPOND":
                # ── Speaker expects a response — short straggler wait then process ──
                print(f"[{ts()}] [EOT] 🟢 RESPOND — waiting {self.STRAGGLER_WAIT}s for stragglers...")
                await asyncio.sleep(self.STRAGGLER_WAIT)

                if self._speaking:
                    print(f"[{ts()}] 🛑 Post-straggler: already speaking — skip")
                    return
                if not self._buffer:
                    print(f"[{ts()}] 🛑 Post-straggler: buffer empty — skip")
                    return

                # Re-read buffer (might have grown during straggler wait)
                result = self._get_buffer_text()
                if not result:
                    return
                spk, full_text, t0 = result

                self._buffer.clear()
                self._partial_text = ""
                self._partial_speaker = ""
                self._vad.end_turn()
                self._last_flushed_text = full_text
                print(f"[{ts()}] 📝 Turn complete: \"{full_text}\"")
                self._start_process(full_text, spk, t0)

            else:
                # ── Speaker still going — long wait, let them finish ──
                print(f"[{ts()}] [EOT] 🟡 WAIT — listening for {self.WAIT_TIMEOUT}s...")
                await asyncio.sleep(self.WAIT_TIMEOUT)

                if self._speaking or not self._buffer:
                    return

                result = self._get_buffer_text()
                if not result:
                    return
                spk, full_text, t0 = result

                self._buffer.clear()
                self._partial_text = ""
                self._partial_speaker = ""
                self._vad.end_turn()
                self._last_flushed_text = full_text
                print(f"[{ts()}] 📝 Wait timeout — processing: \"{full_text}\"")
                self._start_process(full_text, spk, t0)

        except asyncio.CancelledError:
            # New transcript arrived — will schedule a fresh EOT check
            return

    def _get_buffer_text(self) -> tuple[str, str, float] | None:
        """Extract accumulated text from buffer without clearing. Returns (speaker, text, t0) or None."""
        if not self._buffer and not self._partial_text:
            return None

        if self._buffer:
            speaker   = self._buffer[-1][0]
            t0        = self._buffer[0][2]
            full_text = " ".join(txt for _, txt, _ in self._buffer)
            if self._partial_text:
                last_buf = self._buffer[-1][1] if self._buffer else ""
                if self._partial_text not in full_text and self._partial_text != last_buf:
                    full_text = full_text + " " + self._partial_text
        else:
            speaker   = self._partial_speaker or "Unknown"
            t0        = time.time()
            full_text = self._partial_text

        return speaker, full_text, t0

    # ══════════════════════════════════════════════════════════════════════════
    # Process launcher + helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _start_process(self, text, speaker, t0):
        self._generation     += 1
        my_gen                = self._generation
        self._current_text    = text
        self._current_speaker = speaker
        self._interrupt_event.clear()
        # Log user message to conversation history (EOT + process both need this)
        self._convo_history.append(f"{speaker}: {text}")
        self._current_task = asyncio.create_task(self._process(text, speaker, t0, my_gen))

    async def _greet(self, name, t0):
        await asyncio.sleep(1.0)
        if self._speaking:
            return
        greeting = f"Hey {name}, welcome to the call!"
        self._log_sam(greeting)
        await self._speak_simple(greeting, t0)

    def _log_sam(self, text: str):
        self._convo_history.append(f"Sam: {text}")
        self.agent.log_exchange("Sam", text)

    # ══════════════════════════════════════════════════════════════════════════
    # TTS + inject helpers
    # ══════════════════════════════════════════════════════════════════════════

    async def _tts(self, text: str, retries: int = 2) -> bytes:
        """TTS with retry for transient DNS/network errors."""
        last_err = None
        for attempt in range(1 + retries):
            try:
                async with self._tts_semaphore:
                    return await self.speaker._synthesise(text)
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "getaddrinfo" in err_str or "ConnectError" in err_str or "TimeoutError" in err_str:
                    if attempt < retries:
                        wait = 0.5 * (attempt + 1)
                        print(f"[{ts()}] ⚠️  TTS DNS error (attempt {attempt+1}/{1+retries}), retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                raise  # non-network error, don't retry
        raise last_err

    def _combine_audio(self, audio_list: list[bytes]) -> bytes:
        """Combine multiple MP3 audio bytes into one seamless MP3."""
        from pydub import AudioSegment
        import io
        combined = AudioSegment.empty()
        for audio_bytes in audio_list:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            combined += seg
        output = io.BytesIO()
        combined.export(output, format="mp3", bitrate="192k")
        return output.getvalue()

    async def _inject_and_wait(self, audio_bytes: bytes, text: str, label: str, my_gen: int, stop_first: bool = True) -> bool:
        """Inject audio + interruptible playback wait. Returns False if interrupted."""
        if self._interrupt_event.is_set() or my_gen != self._generation:
            return False

        try:
            t_inj = time.time()
            if stop_first:
                try:
                    await self.speaker.stop_audio()
                except Exception:
                    pass

            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await self.speaker._inject_into_meeting(b64)
            self._audio_playing = True
            print(f"[{ts()}] ⏱ Inject {label}: {elapsed(t_inj)}")

            # Calculate actual audio duration instead of guessing
            from Speaker import get_duration_ms
            play_dur = max(500, get_duration_ms(audio_bytes))
            try:
                await asyncio.wait_for(self._interrupt_event.wait(), timeout=play_dur / 1000)
                print(f"[{ts()}] ⚡ Interrupted during {label}")
                self._audio_playing = False
                return False
            except asyncio.TimeoutError:
                pass
            self._audio_playing = False
            return True
        except Exception as e:
            print(f"[{ts()}] ⚠️  Inject failed ({label}): {e}")
            self._audio_playing = False
            return True

    async def _speak_simple(self, text, t0):
        """Simple TTS + inject for greetings etc."""
        if self._speaking:
            return
        self._speaking = True
        try:
            audio = await self._tts(text)
            await self._inject_and_wait(audio, text, "greeting", self._generation)
        except Exception as e:
            print(f"[{ts()}] ⚠️  _speak_simple error: {e}")
        finally:
            self._speaking = False
            self._audio_playing = False

    async def _play_interrupt_ack(self):
        """Play a pre-baked interrupt ack INSTANTLY (no TTS wait)."""
        if not self._interrupt_ack_audio:
            return
        self._interrupt_event.clear()
        self._generation += 1
        self._speaking = True
        try:
            text, audio = random.choice(self._interrupt_ack_audio)
            print(f"[{ts()}] 🙏 Interrupt ack (instant): \"{text}\"")
            await self._inject_and_wait(audio, text, "interrupt-ack", self._generation)
        except Exception as e:
            print(f"[{ts()}] ⚠️  Interrupt ack error: {e}")
        finally:
            self._speaking = False
            self._audio_playing = False

    # ══════════════════════════════════════════════════════════════════════════
    # Background search + TTS preparation
    # ══════════════════════════════════════════════════════════════════════════

    async def _search_and_prepare_audio(self, user_text: str, context: str) -> list[tuple[str, bytes]]:
        """Background: search → summarize → TTS all sentences. Returns ready-to-inject audio."""
        summary = await self.agent.search_and_summarize(user_text, context)

        sentences = self.agent._split_sentences(summary)
        prepared = []
        for sent in sentences:
            try:
                audio = await self._tts(sent)
                prepared.append((sent, audio))
                print(f"[{ts()}] 🔧 Pre-baked TTS: \"{sent[:50]}\"")
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pre-bake TTS failed: {e}")
        return prepared

    async def _deliver_pending(self, my_gen: int):
        """Deliver all pending search results — audio pre-baked, just inject."""
        while self._pending_searches:
            if self._interrupt_event.is_set() or my_gen != self._generation:
                return

            query_text, prepare_task = self._pending_searches.pop(0)
            print(f"[{ts()}] 📬 Delivering pending: \"{query_text[:50]}\"")

            try:
                if not prepare_task.done():
                    prepared = await asyncio.wait_for(prepare_task, timeout=15)
                else:
                    prepared = prepare_task.result()
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pending failed: {e}")
                continue

            if not prepared:
                continue

            prefix = "Oh and about your earlier question."
            try:
                prefix_audio = await self._tts(prefix)
                all_audio = [prefix_audio] + [audio for _, audio in prepared]
                combined_audio = self._combine_audio(all_audio)
                full_text = " ".join(sent for sent, _ in prepared)
                ok = await self._inject_and_wait(combined_audio, f"{prefix} {full_text}", "pending-combined", my_gen)
                if not ok:
                    return
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pending delivery error: {e}")
                continue

            self._log_sam(f"{prefix} {full_text}")
            self.trigger.mark_responded()
            print(f"[{ts()}] ✅ Pending delivered")

    # ══════════════════════════════════════════════════════════════════════════
    # Main processing pipeline
    # ══════════════════════════════════════════════════════════════════════════

    async def _process(self, text, speaker, t0, generation=0):
        if self._speaking:
            print(f"[{ts()}] ⚠️  Already speaking — dropping")
            return

        self._speaking = True
        self._interrupt_event.clear()
        my_gen = generation

        try:
            context = "\n".join(self._convo_history)
            print(f"[{ts()}] 📋 Process context ({len(self._convo_history)} entries): {list(self._convo_history)[-3:] if self._convo_history else '(EMPTY)'}")
            t1 = time.time()
            _active_prepare_task = None
            _active_search_text = text

            # ── Trigger + Router in parallel ─────────────────────────────
            trigger_task = asyncio.create_task(
                self.trigger.should_respond(
                    text, speaker, context,
                    [e["text"] for e in self.agent.rag._entries[-20:]]
                )
            )
            router_task = asyncio.create_task(self.agent._route(text, context))

            print(f"[{ts()}] Trigger + Router in parallel...")
            should = await trigger_task
            print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")

            if not should:
                router_task.cancel()
                return

            route = await router_task
            print(f"[{ts()}] Route: [{route}]")

            # ══════════════════════════════════════════════════════════════
            # [FT] PATH — filler + background search + TTS
            # ══════════════════════════════════════════════════════════════
            if route == "FT":
                prepare_task = asyncio.create_task(
                    self._search_and_prepare_audio(text, context)
                )
                _active_prepare_task = prepare_task
                self._searching = True

                filler = random.choice(FILLERS)
                print(f"[{ts()}] 🗣️ Filler: \"{filler}\"")
                try:
                    filler_audio = await self._tts(filler)
                except Exception as e:
                    print(f"[{ts()}] ⚠️  Filler TTS failed: {e}")
                    filler_audio = None

                if filler_audio:
                    ok = await self._inject_and_wait(filler_audio, filler, "filler", my_gen)
                    if not ok:
                        prepare_task.cancel()
                        print(f"[{ts()}] ⚡ Interrupted during filler — discarding search")
                        return

                try:
                    prepared = await asyncio.wait_for(prepare_task, timeout=15)
                except asyncio.TimeoutError:
                    try:
                        await self._inject_and_wait(
                            await self._tts("Hmm that search is taking too long."),
                            "Hmm that search is taking too long.", "timeout", my_gen
                        )
                    except Exception:
                        pass
                    return
                except asyncio.CancelledError:
                    return

                if self._interrupt_event.is_set() or my_gen != self._generation:
                    return

                if not prepared:
                    return

                full_text = " ".join(sent for sent, _ in prepared)
                combined_audio = self._combine_audio([audio for _, audio in prepared])
                ok = await self._inject_and_wait(combined_audio, full_text, "search-combined", my_gen)

                self._log_sam(full_text)
                self.trigger.mark_responded()
                print(f"[{ts()}] ✅ Done (search)")

            # ══════════════════════════════════════════════════════════════
            # [PM] PATH — stream LLM + parallel TTS + seamless inject
            # ══════════════════════════════════════════════════════════════
            else:
                sentence_queue = asyncio.Queue()
                llm_task = asyncio.create_task(
                    self.agent.stream_sentences_to_queue(text, context, sentence_queue)
                )

                all_sentences: list[str] = []

                # ── Drain ALL sentences from LLM ──
                while True:
                    if self._interrupt_event.is_set() or my_gen != self._generation:
                        llm_task.cancel()
                        return
                    try:
                        item = await asyncio.wait_for(sentence_queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        print(f"[{ts()}] ⚠️  LLM queue timeout")
                        break
                    if item is None:
                        break
                    if item == "__FLUSH__":
                        continue
                    all_sentences.append(item)
                    print(f"[{ts()}] LLM sentence {len(all_sentences)} ({elapsed(t1)}): \"{item}\"")

                if not all_sentences:
                    llm_task.cancel()
                    return

                print(f"[{ts()}] ⏱ LLM complete: {len(all_sentences)} sentence(s) ({elapsed(t1)})")

                # ── TTS ALL sentences in parallel ──
                tts_tasks = [(s, asyncio.create_task(self._tts(s))) for s in all_sentences]
                audio_parts = []
                for s, task in tts_tasks:
                    try:
                        ab = await task
                        audio_parts.append(ab)
                    except Exception as e:
                        print(f"[{ts()}] ⚠️  TTS failed for \"{s[:30]}\": {e}")

                if not audio_parts:
                    return

                print(f"[{ts()}] ⏱ TTS all done: {elapsed(t1)}")

                # ── Combine into ONE MP3 + single inject ──
                if len(audio_parts) == 1:
                    final_audio = audio_parts[0]
                else:
                    final_audio = self._combine_audio(audio_parts)

                from Speaker import get_duration_ms
                total_dur_ms = get_duration_ms(final_audio)

                if self._interrupt_event.is_set() or my_gen != self._generation:
                    return

                ok = await self._inject_and_wait(final_audio, ' '.join(all_sentences), "single-inject", my_gen)
                print(f"[{ts()}] 📊 FIRST AUDIO: {elapsed(t0)}")

                if all_sentences:
                    self._log_sam(' '.join(all_sentences))
                    self.trigger.mark_responded()
                    print(f"[{ts()}] 📊 TOTAL: {elapsed(t0)}")
                    print(f"[{ts()}] ✅ Done (PM)")

        except asyncio.CancelledError:
            print(f"[{ts()}] 🔄 Task cancelled")
            if _active_prepare_task and not _active_prepare_task.done():
                _active_prepare_task.cancel()
        except Exception as e:
            import traceback
            print(f"[{ts()}] ❌ _process error: {e}")
            traceback.print_exc()
        finally:
            self._audio_playing = False
            self._speaking      = False
            self._searching     = False

    # ══════════════════════════════════════════════════════════════════════════
    # Server start
    # ══════════════════════════════════════════════════════════════════════════

    async def start(self):
        self.agent.start()
        await self.speaker.warmup()
        await self._vad.setup()

        # Clear debug file for fresh run (only if debug enabled)
        if os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes"):
            try:
                with open("debug_prompts.txt", "w", encoding="utf-8") as f:
                    f.write(f"=== Debug session started at {ts()} ===\n")
                print(f"[{ts()}] 📝 Debug logging to debug_prompts.txt")
            except Exception:
                pass

        # Pre-bake interrupt ack audio for instant playback
        print(f"[{ts()}] Pre-baking interrupt ack audio...")
        for phrase in _INTERRUPT_ACKS:
            try:
                audio = await self._tts(phrase)
                self._interrupt_ack_audio.append((phrase, audio))
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pre-bake failed for \"{phrase}\": {e}")
        print(f"[{ts()}] ✅ {len(self._interrupt_ack_audio)} interrupt acks pre-baked")

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] WebSocket server ready on ws://0.0.0.0:{self.port}/ws")
        print(f"[{ts()}] Health check: http://localhost:{self.port}/health\n")
