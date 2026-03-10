#!/usr/bin/env python3
import argparse
import io
import json
import math
import os
import struct
import sys
import threading
import wave
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as email_policy_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from order_integration.gemini_stt_pipeline import transcribe_audio_bytes
from order_integration.voice_order_runtime import build_voice_order_runtime


HOME_HTML_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROBOTENDER</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Pretendard:wght@500;700&display=swap");

    :root {
      --bg0: #040b24;
      --bg1: #0a1f54;
      --text: #eef6ff;
      --accent: #59c8ff;
      --accent2: #2a94ff;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 20% 15%, rgba(72, 187, 255, 0.28), transparent 36%),
        radial-gradient(circle at 88% 10%, rgba(58, 138, 255, 0.22), transparent 34%),
        linear-gradient(150deg, var(--bg0), var(--bg1) 70%);
      color: var(--text);
      font-family: "Pretendard", sans-serif;
      padding: 24px;
    }

    .wrap {
      text-align: center;
      width: min(900px, 100%);
      min-height: calc(100vh - 48px);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }

    .kicker {
      margin: 0;
      font-family: "Orbitron", sans-serif;
      letter-spacing: 0.24em;
      font-size: clamp(14px, 2vw, 22px);
      opacity: 0.9;
      color: #9dd8ff;
    }

    .title {
      margin: 0;
      margin-top: 20px;
      font-family: "Orbitron", sans-serif;
      font-weight: 800;
      letter-spacing: 0.06em;
      font-size: clamp(48px, 12vw, 150px);
      line-height: 0.95;
      text-shadow: 0 0 28px rgba(95, 201, 255, 0.42);
    }

    .start-btn {
      margin-top: 36px;
      display: inline-block;
      text-decoration: none;
      border: 0;
      color: #06253f;
      font-weight: 800;
      font-size: clamp(18px, 2.2vw, 28px);
      padding: 14px 36px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      box-shadow: 0 14px 28px rgba(36, 150, 255, 0.36);
      transition: transform 0.15s ease, filter 0.15s ease;
    }

    .start-btn:hover {
      transform: translateY(-1px);
      filter: brightness(1.04);
    }

    .start-btn:active {
      transform: translateY(1px);
    }

    .start-btn.disabled {
      background: linear-gradient(135deg, #7f8da3, #61708a);
      box-shadow: none;
      color: #e8f1ff;
      cursor: not-allowed;
      pointer-events: none;
    }

    .disabled-note {
      margin-top: 16px;
      font-size: 14px;
      color: #b5d0f0;
      opacity: 0.9;
    }
  </style>
</head>
<body>
  <main class="wrap">
    <p class="kicker">VOICE ORDER</p>
    <h1 class="title">ROBOTENDER</h1>
    __START_BUTTON__
    __DISABLED_NOTE__
  </main>
</body>
</html>
"""


ORDER_HTML_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robotender Voice UI</title>
  <style>
    :root {
      --bg-1: #050b2b;
      --bg-2: #0d1b4f;
      --neon: #3aa9ff;
      --neon-strong: #35e0ff;
      --text: #e4f2ff;
      --muted: #b5d0f0;
      --line: rgba(108, 181, 255, 0.35);
      --chip-bg: rgba(8, 28, 81, 0.55);
      --chip-border: rgba(100, 160, 255, 0.45);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Noto Sans KR", sans-serif;
      color: var(--text);
      display: grid;
      place-items: center;
      padding: 20px;
      background:
        radial-gradient(circle at 20% 15%, rgba(54, 116, 255, 0.38), transparent 35%),
        radial-gradient(circle at 80% 20%, rgba(65, 229, 255, 0.26), transparent 32%),
        linear-gradient(170deg, var(--bg-2), var(--bg-1) 72%);
    }

    .panel {
      width: min(740px, 100%);
      border-radius: 20px;
      padding: 28px 22px 24px;
      text-align: center;
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(rgba(14, 42, 115, 0.58), rgba(5, 14, 45, 0.8)),
        radial-gradient(circle at center, rgba(53, 180, 255, 0.12), transparent 60%);
      border: 1px solid var(--line);
      box-shadow:
        0 0 0 1px rgba(112, 187, 255, 0.15) inset,
        0 30px 70px rgba(0, 0, 0, 0.55),
        0 0 45px rgba(65, 180, 255, 0.3);
    }

    .panel::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(0deg, rgba(78, 163, 255, 0.1), rgba(78, 163, 255, 0) 25%),
        repeating-linear-gradient(
          to bottom,
          rgba(125, 190, 255, 0.05) 0,
          rgba(125, 190, 255, 0.05) 1px,
          transparent 1px,
          transparent 6px
        );
      pointer-events: none;
    }

    h1 {
      margin: 6px 0 12px;
      font-size: clamp(30px, 5vw, 46px);
      letter-spacing: -0.02em;
      text-shadow: 0 0 22px rgba(88, 189, 255, 0.45);
    }

    .subtitle {
      margin: 0 auto 18px;
      width: min(540px, 100%);
      font-size: 24px;
      color: var(--muted);
      line-height: 1.55;
    }

    .mic-wrap {
      margin-top: 16px;
      display: flex;
      justify-content: center;
    }

    #mic-btn {
      width: 220px;
      height: 220px;
      border-radius: 50%;
      border: 0;
      cursor: pointer;
      color: #f3fbff;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at center, rgba(17, 38, 115, 0.9), rgba(8, 20, 64, 0.95));
      box-shadow:
        0 0 0 5px rgba(44, 130, 255, 0.25),
        0 0 0 12px rgba(53, 162, 255, 0.14),
        0 0 45px rgba(75, 190, 255, 0.7),
        inset 0 0 30px rgba(38, 169, 255, 0.45);
      transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
      position: relative;
      z-index: 2;
    }

    #mic-btn:hover {
      transform: translateY(-2px) scale(1.02);
      filter: brightness(1.08);
    }

    #mic-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    #mic-btn svg {
      width: 78px;
      height: 78px;
      filter: drop-shadow(0 0 16px rgba(130, 220, 255, 0.7));
    }

    #mic-btn.active {
      animation: pulse 1.5s infinite;
      box-shadow:
        0 0 0 5px rgba(44, 130, 255, 0.45),
        0 0 0 14px rgba(53, 162, 255, 0.2),
        0 0 65px rgba(69, 226, 255, 0.95),
        inset 0 0 40px rgba(86, 215, 255, 0.65);
    }

    .wave {
      position: absolute;
      inset: 50% auto auto 50%;
      width: 290px;
      height: 290px;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      border: 1px solid rgba(118, 210, 255, 0.35);
      pointer-events: none;
      z-index: 1;
      opacity: 0.3;
    }

    #mic-btn.active + .wave {
      animation: spread 1.8s infinite;
    }

    .status {
      margin: 22px 0 8px;
      font-size: 34px;
      font-weight: 700;
      color: #bde8ff;
      text-shadow: 0 0 18px rgba(99, 206, 255, 0.48);
    }

    .preview {
      min-height: 28px;
      margin: 0;
      color: #d8ebff;
      font-size: 17px;
      opacity: 0.95;
      white-space: pre-line;
    }

    .chips {
      margin-top: 20px;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 10px;
    }

    .chip {
      border: 1px solid var(--chip-border);
      background: var(--chip-bg);
      color: #dceeff;
      font-size: 14px;
      padding: 10px 16px;
      border-radius: 999px;
      cursor: pointer;
      transition: 0.2s ease;
    }

    .chip:hover {
      transform: translateY(-1px);
      border-color: rgba(148, 217, 255, 0.85);
      box-shadow: 0 0 14px rgba(97, 200, 255, 0.36);
    }

    .sample-text {
      margin: 0;
      border: 1px solid var(--chip-border);
      background: var(--chip-bg);
      color: #dceeff;
      font-size: 14px;
      padding: 10px 16px;
      border-radius: 999px;
      opacity: 0.95;
      cursor: default;
    }

    .result {
      margin-top: 18px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid rgba(122, 208, 255, 0.5);
      background: rgba(8, 26, 77, 0.5);
      color: #d8ecff;
      font-size: 14px;
      word-break: keep-all;
    }


    .making-overlay {
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(2, 8, 24, 0.72);
      backdrop-filter: blur(2px);
      z-index: 50;
    }

    .making-overlay.show {
      display: grid;
    }

    .making-card {
      width: min(360px, 88vw);
      border: 1px solid rgba(113, 195, 255, 0.42);
      border-radius: 14px;
      background: rgba(8, 24, 66, 0.94);
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
      padding: 20px;
      text-align: center;
      color: #dff1ff;
      font-size: 22px;
      font-weight: 700;
    }

    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.04); }
      100% { transform: scale(1); }
    }

    @keyframes spread {
      0% {
        transform: translate(-50%, -50%) scale(0.86);
        opacity: 0.5;
      }
      100% {
        transform: translate(-50%, -50%) scale(1.18);
        opacity: 0;
      }
    }
  </style>
</head>
<body>
  <main class="panel">
    <h1>안녕하세요, 로보텐더입니다.</h1>
    <p class="subtitle">원하시는 음료를 말씀해 주세요.</p>

    <form method="post" id="order-form">
      <input type="hidden" id="order-text" name="order_text" value="">

      <div class="mic-wrap">
        <button type="button" id="mic-btn" aria-label="음성 인식 토글" title="음성 인식 토글">
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 15a3 3 0 0 0 3-3V7a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm5-3a1 1 0 1 1 2 0 7 7 0 0 1-6 6.93V21h3a1 1 0 1 1 0 2H8a1 1 0 1 1 0-2h3v-2.07A7 7 0 0 1 5 12a1 1 0 1 1 2 0 5 5 0 1 0 10 0Z"/>
          </svg>
        </button>
        <div class="wave"></div>
      </div>

      <p id="status" class="status">Ready</p>
      <p id="preview" class="preview"></p>
      <p id="reply" class="preview"></p>

      <div class="chips">
        <p class="sample-text">소주시킬게</p>
        <p class="sample-text">소맥 말아줘</p>
      </div>
    </form>
  </main>


  <div id="making-overlay" class="making-overlay" aria-live="polite">
    <div class="making-card">제조중입니다.</div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const previewEl = document.getElementById("preview");
    const replyEl = document.getElementById("reply");
    const orderInput = document.getElementById("order-text");
    const micBtn = document.getElementById("mic-btn");
    const chipsEl = document.querySelector(".chips");
    const transcribeUrl = "/stt/transcribe/";
    const ttsUrl = "/tts/";
    const makingOverlay = document.getElementById("making-overlay");

    let mediaRecorder = null;
    let mediaStream = null;
    let chunks = [];
    let recording = false;
    let isMaking = false;
    let currentAudio = null;
    let currentAudioUrl = null;

    let rejectCount = 0;
    let contextMenu = "";

    if (!navigator.mediaDevices || !window.MediaRecorder) {
      statusEl.textContent = "Mic Unsupported";
      previewEl.textContent = "이 브라우저는 음성 녹음을 지원하지 않습니다.";
      micBtn.disabled = true;
    }

    function showMakingOverlay() {
      isMaking = true;
      if (makingOverlay) makingOverlay.classList.add("show");
      statusEl.textContent = "제조중";
      micBtn.disabled = true;
    }

    async function playTts(text) {
      if (!text) return;
      statusEl.textContent = "Speaking...";
      micBtn.disabled = true;

      const formData = new FormData();
      formData.append("text", text);

      const response = await fetch(ttsUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = "TTS failed";
        try {
          const data = await response.json();
          message = data.error || message;
        } catch (_e) {}
        throw new Error(message);
      }

      const audioBlob = await response.blob();

      if (currentAudio) currentAudio.pause();
      if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);

      currentAudioUrl = URL.createObjectURL(audioBlob);
      currentAudio = new Audio(currentAudioUrl);
      await currentAudio.play();
      currentAudio.onended = function () {
        if (!isMaking) {
          statusEl.textContent = "Ready";
          micBtn.disabled = false;
        }
      };
    }

    function handleConfirmed(menuLabel) {
      statusEl.textContent = "주문 완료!";
      replyEl.textContent = menuLabel + " 주문이 확정되었습니다.";
      previewEl.textContent = "";
      micBtn.disabled = false;
      rejectCount = 0;
      contextMenu = "";
    }

    function showForceChoice(options) {
      statusEl.textContent = "메뉴를 선택해 주세요";
      replyEl.textContent = "원하시는 메뉴를 직접 선택해 주세요.";
      micBtn.disabled = true;
      chipsEl.innerHTML = "";
      options.forEach(function (option) {
        const btn = document.createElement("button");
        btn.className = "chip";
        btn.type = "button";
        btn.textContent = option.label;
        btn.addEventListener("click", function () {
          handleConfirmed(option.label);
        });
        chipsEl.appendChild(btn);
      });
    }

    async function sendForTranscription(blob) {
      statusEl.textContent = "Transcribing...";
      let filename = "recording.webm";
      if (blob.type.includes("mp4")) filename = "recording.mp4";
      if (blob.type.includes("mpeg")) filename = "recording.mp3";
      if (blob.type.includes("wav")) filename = "recording.wav";
      const response = await fetch(transcribeUrl, {
        method: "POST",
        headers: {
          "Content-Type": blob.type || "audio/webm",
          "X-Audio-Filename": filename,
          "X-Reject-Count": String(rejectCount),
          "X-Context-Menu": contextMenu || "",
        },
        body: blob,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "STT failed");
      }

      if (data.confirmed) {
        handleConfirmed(data.menu_label);
        return;
      }

      if (data.force_choice) {
        rejectCount++;
        previewEl.textContent = (data.text || "").trim();
        showForceChoice(data.options);
        return;
      }

      const ttsText = (data.tts_text || "").trim();
      if (data.making) {
        showMakingOverlay();
      }
      previewEl.textContent = ttsText || "응답이 없습니다.";
      replyEl.textContent = "";

      if (contextMenu) rejectCount++;
      contextMenu = data.selected_menu || "";

      if (ttsText) {
        await playTts(ttsText);
      } else {
        statusEl.textContent = "Ready";
      }
    }

    async function startRecording() {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
      const selectedType = preferredTypes.find(function (t) {
        return MediaRecorder.isTypeSupported(t);
      });
      mediaRecorder = selectedType
        ? new MediaRecorder(mediaStream, { mimeType: selectedType })
        : new MediaRecorder(mediaStream);
      chunks = [];
      mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) chunks.push(event.data);
      };
      mediaRecorder.onstop = async function () {
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
        if (blob.size < 2048) {
          statusEl.textContent = "Error";
          previewEl.textContent = "녹음이 너무 짧습니다. 1~2초 이상 말씀해 주세요.";
          mediaStream.getTracks().forEach(function (t) { t.stop(); });
          recording = false;
          micBtn.classList.remove("active");
          return;
        }
        mediaStream.getTracks().forEach(function (t) { t.stop(); });
        try {
          await sendForTranscription(blob);
        } catch (error) {
          statusEl.textContent = "Error";
          previewEl.textContent = "음성 인식 오류: " + error.message;
        } finally {
          recording = false;
          micBtn.classList.remove("active");
        }
      };

      recording = true;
      micBtn.classList.add("active");
      statusEl.textContent = "Listening...";
      orderInput.value = "";
      previewEl.textContent = "";
      replyEl.textContent = "";
      mediaRecorder.start(250);
    }

    function stopRecording() {
      if (mediaRecorder && recording && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
    }

    micBtn.addEventListener("click", async function () {
      if (micBtn.disabled) return;
      try {
        if (recording) {
          stopRecording();
        } else {
          await startRecording();
        }
      } catch (error) {
        statusEl.textContent = "Error";
        previewEl.textContent = "마이크 접근 실패: " + error.message;
      }
    });
  </script>
</body>
</html>
"""


DISABLED_HTML_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Robotender Voice UI</title>
  <style>
    body {
      margin: 0;
      padding: 24px;
      min-height: 100vh;
      background: linear-gradient(170deg, #0d1b4f, #050b2b 72%);
      color: #dff1ff;
      font-family: "Noto Sans KR", sans-serif;
      display: grid;
      place-items: center;
    }
    .card {
      width: min(720px, 100%);
      background: rgba(8, 24, 66, 0.94);
      border: 1px solid rgba(113, 195, 255, 0.42);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid #f59e0b;
      color: #f7c46c;
      background: rgba(245, 158, 11, 0.15);
      font-weight: 700;
      margin-bottom: 14px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: currentColor;
      display: inline-block;
    }
    h1 { margin: 0 0 8px; }
    p { margin: 8px 0; line-height: 1.6; color: #b5d0f0; }
    code {
      background: rgba(8, 28, 81, 0.55);
      border: 1px solid rgba(100, 160, 255, 0.45);
      border-radius: 6px;
      padding: 2px 6px;
      color: #dceeff;
    }
  </style>
</head>
<body>
  <section class="card">
    <h1>안녕하세요, 로보텐더입니다.</h1>
    <div class="badge"><span class="dot"></span>사용자 진입 비활성화</div>
    <p>WEB UI 프로세스는 실행 중이지만 현재 주문 진입은 비활성화 상태입니다.</p>
    <p>활성화하려면 <code>VOICE_ORDER_WEBUI_ENABLED=1</code> 또는 <code>--enabled true</code>로 실행하세요.</p>
  </section>
</body>
</html>
"""

ORDER_START_BLOCKED_HTML_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Robotender Voice UI</title>
  <style>
    body {
      margin: 0;
      padding: 24px;
      min-height: 100vh;
      background: linear-gradient(170deg, #0d1b4f, #050b2b 72%);
      color: #dff1ff;
      font-family: "Noto Sans KR", sans-serif;
      display: grid;
      place-items: center;
    }
    .card {
      width: min(720px, 100%);
      background: rgba(8, 24, 66, 0.94);
      border: 1px solid rgba(113, 195, 255, 0.42);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid #f59e0b;
      color: #f7c46c;
      background: rgba(245, 158, 11, 0.15);
      font-weight: 700;
      margin-bottom: 14px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: currentColor;
      display: inline-block;
    }
    h1 { margin: 0 0 8px; }
    p { margin: 8px 0; line-height: 1.6; color: #b5d0f0; }
    code {
      background: rgba(8, 28, 81, 0.55);
      border: 1px solid rgba(100, 160, 255, 0.45);
      border-radius: 6px;
      padding: 2px 6px;
      color: #dceeff;
    }
  </style>
</head>
<body>
  <section class="card">
    <h1>안녕하세요, 로보텐더입니다.</h1>
    <div class="badge"><span class="dot"></span>주문 시작 잠금</div>
    <p>주문 시작 버튼은 현재 비활성화 상태입니다.</p>
    <p>활성화하려면 <code>VOICE_ORDER_WEBUI_ORDER_START_ENABLED=1</code> 또는 <code>--order-start-enabled true</code>로 실행하세요.</p>
  </section>
</body>
</html>
"""


def _is_true_text(value, default=False):
    text = str(value if value is not None else "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _now_text():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _append_progress(server, stage: str, message: str):
    with server.state_lock:
        events = list(getattr(server, "progress_events", []) or [])
        events.append({"ts": _now_text(), "stage": str(stage), "message": str(message)})
        if len(events) > 500:
            events = events[-500:]
        server.progress_events = events
        server.updated_at = _now_text()


def _build_tone_wav(duration_sec: float = 0.28, freq_hz: float = 660.0, sample_rate: int = 16000) -> bytes:
    frames = max(1, int(float(duration_sec) * int(sample_rate)))
    amp = 12000
    with io.BytesIO() as bio:
        with wave.open(bio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(frames):
                value = int(amp * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)))
                wav.writeframes(struct.pack("<h", value))
        return bio.getvalue()


class VoiceOrderWebHandler(BaseHTTPRequestHandler):
    server_version = "VoiceOrderWebUI/1.1"

    def log_message(self, fmt, *args):  # pragma: no cover
        print("[voice-webui] " + (fmt % args))

    def _write_json(self, payload: dict, status_code: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status_code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, html_text: str, status_code: int = 200):
        body = str(html_text).encode("utf-8")
        self.send_response(int(status_code))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_wav(self, wav_bytes: bytes, status_code: int = 200):
        body = bytes(wav_bytes)
        self.send_response(int(status_code))
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", 'inline; filename="synthesized.wav"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_raw_body(self) -> bytes:
        try:
            clen = int(self.headers.get("Content-Length", "0"))
        except Exception:
            clen = 0
        return self.rfile.read(max(0, clen)) if clen > 0 else b""

    def _read_json_body(self):
        raw = self._read_raw_body()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return None

    def _read_audio_payload(self):
        content_type = str(self.headers.get("Content-Type", "") or "")
        raw = self._read_raw_body()
        if not raw:
            return b"", "recording.webm", {}
        filename = str(self.headers.get("X-Audio-Filename", "") or "").strip() or "recording.webm"
        extra_fields = {}

        # Compat: still accept multipart/form-data payloads.
        if content_type.lower().startswith("multipart/form-data"):
            try:
                raw_message = (
                    f"Content-Type: {content_type}\r\n"
                    "MIME-Version: 1.0\r\n\r\n"
                ).encode("utf-8") + raw
                message = BytesParser(policy=email_policy_default).parsebytes(raw_message)
                if message.is_multipart():
                    audio_bytes = b""
                    for part in message.iter_parts():
                        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
                        part_filename = str(part.get_filename() or "").strip()
                        payload = part.get_payload(decode=True) or b""
                        if (field_name == "audio") or part_filename:
                            if payload and (not audio_bytes):
                                audio_bytes = bytes(payload)
                                if part_filename:
                                    filename = part_filename
                        elif field_name:
                            extra_fields[field_name] = payload.decode("utf-8", errors="ignore")
                    if audio_bytes:
                        return audio_bytes, filename, extra_fields
            except Exception:
                pass
        return bytes(raw), filename, extra_fields

    def _state_payload(self):
        with self.server.state_lock:
            return {
                "ok": True,
                "enabled": bool(getattr(self.server, "webui_enabled", False)),
                "order_start_enabled": bool(getattr(self.server, "order_start_enabled", False)),
                "bartender_process_running": bool(getattr(self.server, "bartender_process_running", False)),
                "last_input_text": str(getattr(self.server, "last_input_text", "") or ""),
                "updated_at": str(getattr(self.server, "updated_at", "") or ""),
                "progress_events": list(getattr(self.server, "progress_events", []) or []),
            }

    def _home_html(self, enabled: bool) -> str:
        order_start_enabled = bool(getattr(self.server, "order_start_enabled", False))
        allow_start = bool(enabled and order_start_enabled)
        if allow_start:
            button_html = '<a class="start-btn" href="/order/">주문 시작</a>'
        else:
            button_html = '<button class="start-btn disabled" type="button" aria-disabled="true">주문 시작 준비중</button>'

        if (not enabled):
            note = '<p class="disabled-note">현재 사용자 진입 모드: 비활성화</p>'
        elif not order_start_enabled:
            note = '<p class="disabled-note">주문 시작 버튼은 현재 잠금 상태입니다.</p>'
        else:
            note = ""
        return (
            HOME_HTML_PAGE
            .replace("__START_BUTTON__", button_html)
            .replace("__DISABLED_NOTE__", note)
        )

    def do_GET(self):
        path = str(self.path or "")
        enabled = bool(getattr(self.server, "webui_enabled", False))
        order_start_enabled = bool(getattr(self.server, "order_start_enabled", False))
        can_start_order = bool(enabled and order_start_enabled)
        if path in ("/", "/index.html"):
            self._write_html(self._home_html(enabled))
            return
        if path in ("/order", "/order/"):
            if not enabled:
                self._write_html(DISABLED_HTML_PAGE)
                return
            if not order_start_enabled:
                self._write_html(ORDER_START_BLOCKED_HTML_PAGE)
                return
            self._write_html(ORDER_HTML_PAGE)
            return
        if path.startswith("/api/health"):
            self._write_json(
                {
                    "ok": True,
                    "service": "voice-order-webui",
                    "enabled": enabled,
                    "order_start_enabled": order_start_enabled,
                    "can_start_order": can_start_order,
                }
            )
            return
        if path.startswith("/api/state"):
            self._write_json(self._state_payload())
            return
        self._write_json({"ok": False, "error": "not_found"}, status_code=404)

    def do_POST(self):
        path = str(self.path or "")
        can_start_order = bool(
            bool(getattr(self.server, "webui_enabled", False))
            and bool(getattr(self.server, "order_start_enabled", False))
        )

        if path == "/stt/transcribe/":
            if not can_start_order:
                self._write_json({"error": "service_disabled"}, status_code=503)
                return
            audio_bytes, filename, _ = self._read_audio_payload()
            if not audio_bytes:
                self._write_json({"error": "audio file is required"}, status_code=400)
                return
            if len(audio_bytes) < 2048:
                self._write_json({"error": "audio too short; please record a bit longer"}, status_code=400)
                return
            _append_progress(self.server, "input", f"사용자 음성 입력 수신(filename={filename}, bytes={len(audio_bytes)})")
            try:
                stt = transcribe_audio_bytes(audio_bytes, filename=filename)
            except Exception as exc:
                msg = str(exc) or "stt_transcribe_failed"
                _append_progress(self.server, "stt", f"Gemini STT 실패: {msg}")
                self._write_json({"error": msg}, status_code=500)
                return

            transcript = str(stt.text or "").strip()
            emotion = str(stt.emotion or "").strip()
            recommend_menu = str(stt.recommend_menu or "").strip()
            reason = str(stt.reason or "").strip()

            if not transcript:
                _append_progress(self.server, "stt", "Gemini STT 결과 비어있음")
                with self.server.state_lock:
                    self.server.last_input_text = ""
                    self.server.updated_at = _now_text()
                self._write_json(
                    {
                        "text": "",
                        "transcript": "",
                        "emotion": emotion,
                        "recommend_menu": recommend_menu,
                        "reason": reason,
                        "tts_text": "다시 한번 말씀해주시겠어요?",
                        "status": "retry",
                        "selected_menu": "",
                        "selected_menu_label": "",
                        "recipe": {},
                        "route": "stt_empty",
                        "making": False,
                    }
                )
                return

            runtime = build_voice_order_runtime(
                input_text=transcript,
                recommend_menu=recommend_menu,
                allow_llm=True,
                emotion=emotion,
                reason=reason,
            )
            result = dict(runtime.result_payload or {})

            with self.server.state_lock:
                self.server.last_input_text = transcript
                self.server.bartender_process_running = True
                self.server.updated_at = _now_text()
            _append_progress(self.server, "stt", f"Gemini STT 결과: {transcript}")
            for event in list(runtime.events or []):
                if not isinstance(event, dict):
                    continue
                stage = str(event.get("stage", "pipeline") or "pipeline")
                message = str(event.get("message", "") or "")
                data = event.get("data")
                if data is not None:
                    try:
                        message = f"{message} | {json.dumps(data, ensure_ascii=False)}"
                    except Exception:
                        message = f"{message} | {data}"
                _append_progress(self.server, stage, message)
            self._write_json(
                {
                    "text": transcript,
                    "transcript": transcript,
                    "emotion": emotion,
                    "recommend_menu": recommend_menu,
                    "reason": reason,
                    "tts_text": str(result.get("tts_text", "") or ""),
                    "status": str(result.get("status", "") or ""),
                    "selected_menu": str(result.get("selected_menu", "") or ""),
                    "selected_menu_label": str(result.get("selected_menu_label", "") or ""),
                    "recipe": result.get("recipe", {}) or {},
                    "route": str(result.get("route", "") or ""),
                    "making": False,
                }
            )
            return

        if path == "/tts/":
            if not can_start_order:
                self._write_json({"error": "service_disabled"}, status_code=503)
                return
            _ = self._read_raw_body()
            self._write_wav(_build_tone_wav())
            return

        payload = self._read_json_body()
        if payload is None:
            self._write_json({"ok": False, "error": "invalid_json"}, status_code=400)
            return

        if path == "/api/control/input":
            text = str(payload.get("input_text", "") or "").strip()
            with self.server.state_lock:
                self.server.last_input_text = text
                self.server.updated_at = _now_text()
            _append_progress(self.server, "input", f"사용자 입력 갱신: {text if text else '-'}")
            self._write_json(self._state_payload())
            return

        if path == "/api/control/start":
            if not can_start_order:
                self._write_json(
                    {
                        "ok": False,
                        "error": "service_disabled",
                        "message": "voice order start is disabled",
                    },
                    status_code=503,
                )
                return
            text = str(payload.get("input_text", "") or "").strip()
            with self.server.state_lock:
                self.server.bartender_process_running = True
                if text:
                    self.server.last_input_text = text
                self.server.updated_at = _now_text()
            _append_progress(self.server, "system", "사용자가 주문 처리/실행 요청")
            if text:
                _append_progress(self.server, "input", f"입력 반영: {text}")
            _append_progress(self.server, "pipeline", "음성처리 백엔드 연동 준비 단계(미연동)")
            self._write_json(self._state_payload())
            return

        if path == "/api/control/stop":
            with self.server.state_lock:
                self.server.bartender_process_running = False
                self.server.updated_at = _now_text()
            _append_progress(self.server, "system", "사용자가 중지 요청")
            self._write_json(self._state_payload())
            return

        if path == "/api/control/clear":
            with self.server.state_lock:
                self.server.progress_events = []
                self.server.updated_at = _now_text()
            self._write_json(self._state_payload())
            return

        self._write_json({"ok": False, "error": "not_found"}, status_code=404)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Voice order WEB UI server")
    parser.add_argument("--host", default=str(os.environ.get("VOICE_ORDER_WEBUI_HOST", "0.0.0.0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VOICE_ORDER_WEBUI_PORT", "8000")))
    parser.add_argument("--enabled", default=str(os.environ.get("VOICE_ORDER_WEBUI_ENABLED", "0")))
    parser.add_argument(
        "--order-start-enabled",
        default=str(os.environ.get("VOICE_ORDER_WEBUI_ORDER_START_ENABLED", "0")),
    )
    return parser.parse_args(argv)


def main(argv=None):
    if load_dotenv is not None:
        project_root = SRC_ROOT.parent
        load_dotenv(dotenv_path=project_root / ".env", override=False)

    args = parse_args(argv)
    host = str(args.host or "0.0.0.0")
    port = max(1, int(args.port))
    enabled = bool(_is_true_text(args.enabled, default=False))
    order_start_enabled = bool(_is_true_text(args.order_start_enabled, default=False))
    server = ThreadingHTTPServer((host, port), VoiceOrderWebHandler)
    server.webui_enabled = enabled
    server.order_start_enabled = order_start_enabled
    server.state_lock = threading.Lock()
    server.bartender_process_running = False
    server.last_input_text = ""
    server.progress_events = []
    server.updated_at = _now_text()

    _append_progress(
        server,
        "system",
        (
            "WEB UI 서버 시작("
            f"host={host}, port={port}, "
            f"enabled={'on' if enabled else 'off'}, "
            f"order_start={'on' if order_start_enabled else 'off'}"
            ")"
        ),
    )

    print(
        (
            f"[voice-webui] listening on http://{host}:{port} "
            f"(enabled={'on' if enabled else 'off'}, "
            f"order_start={'on' if order_start_enabled else 'off'})"
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
