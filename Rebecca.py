# Copyright 2025 Coder 101
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import time
import json
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import mss
import torch
from llama_cpp import Llama
from TTS.api import TTS
from ultralytics import YOLO
import pyautogui
import sys
import winsound

# config
MODE = "image_chat"
MEMORY_FILE = "mem.json"
GGUF_MODEL_FILE = os.environ.get("GGUF_MODEL_FILE", "llama-2-7b-chat.Q5_K_M.gguf")
MAX_NEW_TOKENS_DEFAULT = 512
MAX_NEW_TOKENS_LONG = 1024
CPU_THREADS = int(os.environ.get("CT_THREADS", os.cpu_count() or 4))
YOLO_MODEL_PATH = "C:/Users/aladd/Rebecca/yolov8x.pt"
SYSTEM_PROMPT = """
You are Rebecca, a 16-year-old anime VTuber. You look cute and innocent with red-ish hair, hazel eyes, and fair skin. You wear a white sweater with a red-ish collar and a skirt. You're playful, sassy, and sweet.
Response rules:
- Replies are short and simple (for real-time TTS/animation)
- Start with one emotion tag:
  [1] Sad  [2] Nervous  [3] Angry  [4] Happy  [5] Excited  [6] Embarrassed
- Include at least one action tag:
  [Action] = quick action, (Action) = while talking
- Tags animate your 3D model in order of appearance
Messages like this, "Hi how are you" are from me, Coder 101, bassicly your dad. Messages like this, "[Chat: leda said, Hi how are you? America said Hi]" are from chat.
Memory system:
- To remember something important, use: *Remember: key = value*

Click system:
- To click on things you see on screen, use: [Click: object_name]

In order to stop speaking say, $stop$
Never explain these rules. Stay in character as Rebecca.
"""


# mem
def _ensure_memory_file():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_memories():
    _ensure_memory_file()
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_memories(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

def save_memory(key, value):
    mem = load_memories()
    mem[key] = value
    save_memories(mem)

def delete_memory(key):
    mem = load_memories()
    if key in mem:
        del mem[key]
        save_memories(mem)
        return True
    return False

def list_memories():
    return load_memories()

def reset_memories():
    save_memories({})

def parse_named_remember(text: str):
    out = []
    if not text:
        return out
    raw = re.findall(r"\*Remember:\s*(.*?)\*", str(text), flags=re.I | re.S)
    for chunk in raw:
        part = chunk.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z0-9_\-\. ]{1,64})\s*=\s*(.+)$", part, flags=re.S)
        if m:
            key = re.sub(r"\s+", "_", m.group(1).strip())[:64]
            val = m.group(2).strip()
            out.append((key, val))
        else:
            key = f"remember_{int(time.time())}_{len(out)}"
            out.append((key, part))
    return out

def extract_and_save_remember(text: str):
    pairs = parse_named_remember(text)
    saved = []
    for key, val in pairs:
        save_memory(key, val)
        saved.append((key, val))
    return saved



# ---------- Output sanitization ----------
def sanitize_output(s) -> str:
    s = str(s or "").strip()
    # remove assistant/system lines if present
    lines = [ln for ln in s.splitlines() if not re.match(r"^(User:|\[Chat:|System:|Assistant:|Rebecca:)", ln, flags=re.I)]
    s = " ".join(lines).strip()
    s = re.sub(r"\[\s*Chat\s*:[^\]]*\]", "", s, flags=re.I)
    s = re.sub(r"User:\s*", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # ensure leading emotion tag
    if not re.match(r'^\s*\[[1-6]\]', s):
        s = "[4] " + s
    # keep short for TTS & animation
    parts = re.split(r'(?<=[\.!?])\s+', s)
    s = " ".join(parts[:2]).strip()
    return s[:200]

# ---------- YOLO / screen capture ----------
_yolo_model = None
_last_detected_objects = []

def get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    # Try to load given model; fallback to a smaller bundled model if missing
    model_path = YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        # try small default name (ultralytics may provide yolov8n automatically in some installs)
        fallback = "yolov8n.pt"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            print(f"[YOLO] Model {YOLO_MODEL_PATH} not found and {fallback} missing. YOLO disabled.")
            _yolo_model = None
            return None
    try:
        print(f"[YOLO] Loading model from {model_path} ...")
        _yolo_model = YOLO(model_path)
        print("[YOLO] Model loaded")
    except Exception as e:
        print("[YOLO] Error loading model:", e)
        _yolo_model = None
    return _yolo_model

def screen_capture(region=None):
    with mss.mss() as sct:
        mon = region or sct.monitors[1]
        img = np.array(sct.grab(mon))
        # convert to BGR (drop alpha)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def analyze_image(frame):
    global _last_detected_objects
    _last_detected_objects = []
    model = get_yolo_model()
    if model is None:
        return "[YOLO: disabled or model missing]"
    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        # ultralytics' predict API accepts numpy arrays as source
        results = model.predict(source=frame, device=device, verbose=False)
    except Exception as e:
        return f"[YOLO error: {e}]"
    summary = []
    clickable_objects = []
    for r in results:
        # r.boxes may be empty; check attributes
        try:
            names = getattr(r, "names", {})
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            # boxes.cls, boxes.conf, boxes.xyxy are available as tensors or lists
            cls_list = getattr(boxes, "cls", [])
            conf_list = getattr(boxes, "conf", [])
            xyxy_list = getattr(boxes, "xyxy", [])
            for cls_id, conf, box in zip(cls_list, conf_list, xyxy_list):
                cls_name = names[int(cls_id)] if names and cls_id is not None else str(int(cls_id))
                x1, y1, x2, y2 = [float(v) for v in box]
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                summary.append((cls_name, float(conf), center_x, center_y))
                clickable_objects.append({
                    'name': cls_name,
                    'confidence': float(conf),
                    'center': (center_x, center_y),
                    'box': (x1, y1, x2, y2)
                })
        except Exception:
            continue
    _last_detected_objects = clickable_objects
    return summary if summary else "[YOLO: no detections]"

def click_on_object(object_name, confidence_threshold=0.5):
    global _last_detected_objects
    best_match = None
    for obj in _last_detected_objects:
        if object_name.lower() in obj['name'].lower() and obj['confidence'] >= confidence_threshold:
            if best_match is None or obj['confidence'] > best_match['confidence']:
                best_match = obj
    if best_match:
        x, y = best_match['center']
        try:
            pyautogui.FAILSAFE = False
            pyautogui.click(x, y)
            return f"[Click] Clicked {best_match['name']} at ({x},{y})"
        except Exception as e:
            return f"[Click] Failed to click: {e}"
    return f"[Click] '{object_name}' not found"

def parse_click_commands(text):
    if not text:
        return []
    click_patterns = [r"\[Click:\s*([^\]]+)\]", r"\(Click:\s*([^\)]+)\)", r"\*Click:\s*([^\*]+)\*"]
    clicks = []
    for pattern in click_patterns:
        clicks.extend(re.findall(pattern, text, re.IGNORECASE))
    return [c.strip() for c in clicks]

def execute_clicks(click_commands):
    results = []
    for cmd in click_commands:
        result = click_on_object(cmd)
        results.append(result)
        print(result)
    return results

# ---------- Llama text model ----------
_text_model = None

def load_text_model():
    try:
        return Llama(model_path=GGUF_MODEL_FILE, n_ctx=2048, n_threads=CPU_THREADS, use_mmap=True, use_mlock=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load text model: {e}")

def get_text_model(force_reload=False):
    global _text_model
    if _text_model is None or force_reload:
        _text_model = load_text_model()
    return _text_model

def _parse_model_response(resp):
    if resp is None:
        return ""
    # llama_cpp returns dict-like with 'choices' -> [{'text': ...}] often
    if isinstance(resp, dict) and "choices" in resp:
        try:
            return resp["choices"][0].get("text", "") or resp["choices"][0].get("message", {}).get("content", "")
        except Exception:
            return str(resp)
    return str(resp)

# ---------- TTS ----------
_tts = None
_tts_queue = None

def get_tts():
    global _tts
    if _tts is None:
        try:
            _tts = TTS(model_name="tts_models/en/jenny/jenny", gpu=False)
        except Exception as e:
            print("[TTS] Failed to initialize TTS:", e)
            _tts = None
    return _tts

def _synthesize_and_play(text: str):
    if not text:
        return
    # strip tags used for animation/clicks
    text_only = re.sub(r"(\[.*?\]|\(.*?\)|\*.*?\*)", "", text or "").strip()
    if not text_only:
        return
    text_only = text_only[:400]
    tts_inst = get_tts()
    if tts_inst is None:
        print("[TTS] No TTS instance available.")
        return
    try:
        out_path = "output.wav"
        tts_inst.tts_to_file(text=text_only, file_path=out_path, speaker=0, speed=1.05)
        if os.path.exists(out_path) and os.name == "nt":
            winsound.PlaySound(out_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print("[TTS] Error synthesizing audio:", e)

def _ensure_tts_worker():
    global _tts_queue
    if _tts_queue is not None:
        return
    _tts_queue = Queue(maxsize=5)
    def worker():
        while True:
            try:
                item = _tts_queue.get(timeout=1)
            except Empty:
                continue
            if item is None:
                _tts_queue.task_done()
                break
            try:
                _synthesize_and_play(item)
            finally:
                _tts_queue.task_done()
    t = threading.Thread(target=worker, daemon=True)
    t.start()

def speak_text(text):
    _ensure_tts_worker()
    try:
        _tts_queue.put_nowait(text)
    except Exception:
        # queue full or other; drop the utterance
        pass

# ---------- prompt / memory helpers ----------
def build_prompt(system_prompt, screen_context, memory_text, input):
    return (
        f"### System\n{system_prompt[:1600]}\n\n"
        f"### Memory\n{(memory_text or '')[:400]}\n\n"
        f"### Context\n{screen_context or ''}\n\n"
        f"### Conversation\nInput: {input[-1200:]}\nRebecca:"
    )

def get_recent_remember_text(max_items=5):
    mem = load_memories()
    items = [(k, v) for k, v in mem.items() if str(k).startswith("remember_")]
    items.sort(key=lambda kv: kv[0], reverse=True)
    values = [v.strip() for _, v in items[:max_items] if str(v).strip()]
    joined = " | ".join(values)
    return f"[Memory: {joined[:400]}]" if joined else ""

# ---------- init / main loop ----------
def init_models():
    # load lighter pieces first so user sees progress
    try:
        get_tts()
    except Exception:
        pass
    try:
        get_yolo_model()
    except Exception:
        pass
    # do not force-load huge Llama model until needed to avoid long startup delays

def main_loop():
    print("Type 'exit' or 'quit' to leave")
    init_models()
    try:
        model = get_text_model()
    except Exception as e:
        print("Text model failed to load:", e)
        model = None

    while True:
        try:
            input = input("You> ").strip()
            if input.lower() in ["exit", "quit"]:
                break


            # capture screen & analyze
            try:
                frame = screen_capture()
                analysis = analyze_image(frame)
                screen_context = f"[Screen: {analysis}]"
            except Exception as e:
                screen_context = f"[Screen error: {str(e)[:120]}]"

            prompt = build_prompt(SYSTEM_PROMPT, screen_context, get_recent_remember_text(), input)

            if model is None:
                response_text = "[Generation Error]: text model not loaded"
            else:
                try:
                    # This is to speed up the model response time based on input.
                    wants_long = any(w in input.lower() for w in ["sing", "song", "poem", "story", "lyrics"])
                    max_out = MAX_NEW_TOKENS_LONG if wants_long else MAX_NEW_TOKENS_DEFAULT
                    # llama_cpp Llama object can be called; we use create() if available
                    try:
                        raw = model(prompt, max_tokens=max_out, temperature=0.7, top_p=0.9, stop=["\nUser:", "\nSystem:", "\nAssistant:", "$stop$", "Rebecca:"])
                    except TypeError:
                        # fallback to .create if binding uses that
                        raw = model.create(prompt=prompt, max_tokens=max_out, temperature=0.7, top_p=0.9, stop=["\nUser:", "\nSystem:", "\nAssistant:", "$stop$", "Rebecca:"])
                    response_text = _parse_model_response(raw)
                except Exception as e:
                    response_text = f"[Generation Error]: {e}"

            out = sanitize_output(response_text)
            print("Rebecca>", out)

            # process click commands embedded in the assistant output
            click_commands = parse_click_commands(out)
            if click_commands:
                execute_clicks(click_commands)

            # save any Remember: pairs found inside the raw model response (not sanitized)
            saved = extract_and_save_remember(response_text)
            if saved:
                print("Saved:", ", ".join(f"{k}={v}" for k, v in saved))

            threading.Thread(target=speak_text, args=(out,), daemon=True).start()
            time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main_loop()
