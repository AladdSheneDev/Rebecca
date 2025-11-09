# Copyright 2025 Viewer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License, is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# I had fun with emojies thanks to linux!
import os
import re
import time
import json
import threading
from queue import Queue, Empty
import subprocess
import sys
import glob
import shutil
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import collections
import torch
import collections
from llama_cpp import Llama
import edge_tts
import asyncio
import soundfile as sf
from ultralytics import YOLO

# config
NOTES = input("Notes for this stream:")
MODE = "image_chat"
MEMORY_FILE = "mem.json"
GGUF_MODEL_FILE = os.environ.get("GGUF_MODEL_FILE", "Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf")
MAX_NEW_TOKENS_DEFAULT = 3086
MAX_NEW_TOKENS_LONG = 5026
CPU_THREADS = int(os.environ.get("CT_THREADS", os.cpu_count() or 4))
YOLO_MODEL_PATH = "yolov8x.pt"
KEEP_SCREENSHOTS = 10  # Keep last 10 screenshots so I can see what goes on in Rebecca's head.
# All those comments below I wrote manualy. SOS
# Emotion-to-speed mapping (pitch removed - sounds demonic!)
emotion_speed_map = {
        1: 1.0,   # admiration
        2: 0.95,  # adoration (slightly slower, tender)
        3: 1.0,   # aesthetic pleasure
        4: 1.0,   # appreciation (default)
        5: 1.3,   # amusement (faster, laughing)
        6: 0.9,   # anger (slower, intense)
        7: 1.1,   # anxiety (slightly faster, nervous)
        8: 0.95,  # awe (slower, reverent)
        9: 1.05,  # awkwardness (slightly faster)
        10: 0.85, # boredom (slower, monotone)
        11: 0.9,  # calmness (slower, peaceful)
        12: 1.05, # confusion (slightly faster)
        13: 1.0,  # craving
        14: 0.9,  # disgust (slower)
        15: 0.9,  # empathic pain (slower)
        16: 0.95, # entrancement (slower, mesmerized)
        17: 1.4,  # excitement (very fast!)
        18: 1.1,  # fear (faster, panicked)
        19: 1.15, # horror (faster, shocked)
        20: 1.05, # interest (slightly faster)
        21: 1.25, # joy (faster, happy)
        22: 0.95, # love (slightly slower, tender)
        23: 1.0,  # relief
        24: 0.9,  # romance (slower, intimate)
        25: 0.8,  # sadness (slower, melancholic)
        26: 1.0,  # satisfaction
        27: 1.2,  # triumph (faster, victorious)
        28: 0.95, # content/just existing (slightly slower)
        29: 1.1,  # mild embarrassment (slightly faster)
        30: 1.15, # playful mischief (faster, teasing)
        31: 1.05, # curiosity (slightly faster)
        32: 0.7,  # sleepiness (very slow, drowsy)
        33: 0.9,  # reflective/thoughtful (slower)
    }

torch.serialization.add_safe_globals([
    'collections.OrderedDict',
])

# Bassicly tell her what the stream is about.
if (NOTES == ""):
    NOTES = "No notes today"
else:
    NOTES = f"Notes for this stream: {NOTES}"
try:
    yolo = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"[YOLO] Error loading model '{YOLO_MODEL_PATH}': {e!r}")
    yolo = None

torch.serialization.add_safe_globals([
    'collections.OrderedDict',
])

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
# This is the most important part of Rebeccca, she likes to make up conversations.
def sanitize_output(s) -> str:
    s = str(s or "").strip()

    cutoff_patterns = [
        r'\n\s*Rebecca:',  
        r'\n\s*rebecca:',      
        r'\n\s*Viewer:',   
        r'\n\s*Viewer:',    
        r'\s+Rebecca:',  
        r'\s+Viewer:',   
    ]

    earliest_cutoff = len(s)
    for pattern in cutoff_patterns:
        match = re.search(pattern, s, flags=re.I)
        if match:
            earliest_cutoff = min(earliest_cutoff, match.start())

    if earliest_cutoff < len(s):
        s = s[:earliest_cutoff].strip()

    s = re.sub(r"\[\s*Chat\s*:[^\]]*\]", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()

    if not re.match(r'^\s*\[\d+\]', s):
        # no emotion detected, default to appreciation
        s = "[4] " + s

    return s

# The eyes that see it all...
_yolo_model = None
_last_detected_objects = []

def _find_possible_yolo_paths():
    candidates = []
    if YOLO_MODEL_PATH:
        candidates.append(YOLO_MODEL_PATH)
    # common fallbacks (local only)
    candidates += ["yolov8x.pt", "yolov8n.pt", "yolov8s.pt"]
    if os.path.isdir("models"):
        for name in os.listdir("models"):
            if name.endswith(".pt") or name.endswith(".pth"):
                candidates.append(os.path.join("models", name))
    # dedupe while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    candidate_paths = _find_possible_yolo_paths()
    found = None
    for p in candidate_paths:
        if os.path.exists(p):
            found = p
            break

    if found is None:
        print(f"[YOLO] No local model found among {candidate_paths}. YOLO disabled (no auto-download).")
        _yolo_model = None
        return None

    model_path = found
    try:
        print(f"[YOLO] Loading model from {model_path} ...")

        # PyTorch 2.6+ compatibility: temporarily patch torch.load to allow pickle
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            yolo = YOLO(model_path)
            yolo.fuse()  # Optimize model
            print(f"[YOLO] ✅ Model loaded successfully!")
        finally:
            torch.load = original_load  # Restore original

    except Exception as e:
        print(f"[YOLO] ❌ Error loading model '{model_path}': {e!r}")
        print("[YOLO] Rebecca is blind - click features disabled")
        yolo = None

    _yolo_model = yolo
    return _yolo_model

def screen_capture(region=None):
    """Cross-desktop screen capture with GNOME/Wayland/X11 fallbacks."""
    try:
        screenshots_dir = os.path.join(os.path.dirname(__file__) or ".", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = os.path.join(screenshots_dir, f"rebecca_view_{timestamp}.png")
        latest_file = os.path.join(screenshots_dir, "latest.png")

        # Detect environment
        is_wayland = os.environ.get('WAYLAND_DISPLAY') is not None
        is_gnome = 'gnome' in os.environ.get('XDG_CURRENT_DESKTOP', '').lower()

        methods = []
        # GNOME 49 Wayland: flameshot works, gnome-screenshot captures black screens
        if is_gnome and is_wayland:
            methods.append(('flameshot', ['flameshot', 'full', '-p', temp_file]))
        if is_gnome and not is_wayland:
            methods.append(('gnome-screenshot', ['gnome-screenshot', '-f', temp_file]))
            methods.append(('spectacle', ['spectacle', '-b', '-n', '-o', temp_file]))
        if is_wayland and not is_gnome:
            methods.append(('grim', ['grim', temp_file]))
            methods.append(('grimshot', ['grimshot', 'save', 'screen', temp_file]))
        # Fallbacks for X11 or as last resort
        methods.append(('flameshot', ['flameshot', 'full', '-p', temp_file]))
        methods.append(('import', ['import', '-window', 'root', temp_file]))
        methods.append(('scrot', ['scrot', temp_file]))

        success = False
        last_error = None
        for method_name, cmd in methods:
            try:
                print(f"[Screen] Trying {method_name}...")
                subprocess.run(cmd, check=True, capture_output=True, timeout=8)
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    print(f"[Screen] Success with {method_name}!")
                    success = True
                    break
                else:
                    print(f"[Screen] {method_name} produced empty file")
            except FileNotFoundError:
                print(f"[Screen] {method_name} not found")
                continue
            except subprocess.TimeoutExpired:
                print(f"[Screen] {method_name} timed out")
                last_error = f"{method_name} timed out"
                continue
            except subprocess.CalledProcessError as e:
                print(f"[Screen] {method_name} failed: {e}")
                last_error = str(e)
                continue
            except Exception as e:
                print(f"[Screen] {method_name} error: {e}")
                last_error = str(e)
                continue

        if not success:
            raise Exception(f"All screenshot methods failed. Last error: {last_error}")

        shutil.copy(temp_file, latest_file)
        screenshot_files = sorted(glob.glob(os.path.join(screenshots_dir, "rebecca_view_*.png")))
        if len(screenshot_files) > KEEP_SCREENSHOTS:
            for old_file in screenshot_files[:-KEEP_SCREENSHOTS]:
                try:
                    os.remove(old_file)
                except Exception:
                    pass

        # Load image
        img = Image.open(temp_file).convert("RGB")
        img_array = np.array(img)

        # Convert RGB to BGR for YOLO cause computer vision is complex (sarcasm)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return img_bgr
    except Exception as e:
        print(f"[Screen] Capture error: {e}")
        raise

def analyze_image(frame):
    global _last_detected_objects
    _last_detected_objects = []
    model = get_yolo_model()
    if model is None:
        return "[YOLO: disabled or model missing]"
    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        h, w = frame.shape[:2]
        imgsz = 640
        print(f"[YOLO] Analyzing at {imgsz}x{imgsz} (original: {w}x{h})")
        results = model.predict(source=frame, device=device, verbose=False, imgsz=imgsz, conf=0.5, iou=0.45)
    except Exception as e:
        return f"[YOLO error: {e}]"

    summary = []
    clickable_objects = []
    try:
        for r in results:
            try:
                names = getattr(r, "names", None) or {}
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                # 67676767676767676767676767676776767676767676767676767676767676767676767676767676766776767676767676767676767676767676767676767667
                cls_list = []
                conf_list = []
                xyxy_list = []

                # boxes may present attributes differently depending on version
                if hasattr(boxes, "cls"):
                    try:
                        cls_np = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                        cls_list = list(cls_np)
                    except Exception:
                        # if it ever gets to this point, i was wrong 
                        cls_list = list(getattr(boxes, "cls", [])) or []
                else:
                    cls_list = list(getattr(boxes, "classes", [])) or []

                if hasattr(boxes, "conf"):
                    try:
                        conf_np = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                        conf_list = list(conf_np)
                    except Exception:
                        conf_list = list(getattr(boxes, "conf", [])) or []
                else:
                    conf_list = list(getattr(boxes, "scores", [])) or []

                if hasattr(boxes, "xyxy"):
                    try:
                        xyxy_np = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                        xyxy_list = list(xyxy_np)
                    except Exception:
                        xyxy_list = list(getattr(boxes, "xyxy", [])) or []
                else:
                    xyxy_list = list(getattr(boxes, "boxes", [])) or []

                for idx in range(min(len(cls_list), len(xyxy_list))):
                    try:
                        cid = int(cls_list[idx])
                    except Exception:
                        try:
                            cid = int(float(cls_list[idx]))
                        except Exception:
                            cid = 0
                    conf = float(conf_list[idx]) if idx < len(conf_list) else 0.0
                    box = xyxy_list[idx]
                    if box is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cls_name = names[int(cid)] if names and int(cid) in names else str(int(cid))
                    summary.append((cls_name, float(conf), center_x, center_y))
                    clickable_objects.append({
                        'name': cls_name,
                        'confidence': float(conf),
                        'center': (center_x, center_y),
                        'box': (x1, y1, x2, y2)
                    })
            except Exception:
                continue
    except Exception as e:
        return f"[YOLO parse error: {e}]"

    _last_detected_objects = clickable_objects
    return summary if summary else "[YOLO: no detections]"

def click_on_object(object_name, confidence_threshold=0.5):
    global _last_detected_objects
    best_match = None
    for obj in _last_detected_objects:
        try:
            if object_name.lower() in obj['name'].lower() and obj['confidence'] >= confidence_threshold:
                if best_match is None or obj['confidence'] > best_match['confidence']:
                    best_match = obj
        except Exception:
            continue
    if best_match:
        x, y = best_match['center']
        try:
            # Due to linux I am using ydotool to click the mouse, if your on Windows, fuck you!
            subprocess.run(['ydotool', 'mousemove', '--absolute', str(x), str(y)], check=True)
            time.sleep(0.05)
            subprocess.run(['ydotool', 'click', '0xC0'], check=True)
            return f"[Click] Clicked {best_match['name']} at ({x},{y})"
        except subprocess.CalledProcessError as e:
            return f"[Click] Failed to click: {e}"
        except FileNotFoundError:
            return "[Click] ydotool not found."
        except Exception as e:
            return f"[Click] Unexpected click error: {e}"
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
    global _last_detected_objects

    # Don't try clicking if Rebecca can't see anything
    if not _last_detected_objects:
        print("[Click] No objects detected ")
        return []

    results = []
    for cmd in click_commands:
        result = click_on_object(cmd)
        results.append(result)
        print(result)
    return results

# El ze brain.
_text_model = None

def load_text_model():
    try:
        # raise helpful error if file doesn't exist
        if not os.path.exists(GGUF_MODEL_FILE):
            raise FileNotFoundError(f"GGUF model file not found: {GGUF_MODEL_FILE}")
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
    if isinstance(resp, dict) and "choices" in resp:
        try:
            return resp["choices"][0].get("text", "") or resp["choices"][0].get("message", {}).get("content", "")
        except Exception:
            return str(resp)
    if hasattr(resp, "get") and callable(getattr(resp, "get")):
        try:
            return resp.get("choices", [{}])[0].get("text", "") or ""
        except Exception:
            pass
    return str(resp)

# So the TTS is massive...
_tts = None
_tts_queue = None

def get_tts():
    global _tts
    if _tts is None:
        try:
            print("[TTS] Initializing TTS...")
            # edge-tts doesn't need initialization, just put that here for fun.
            _tts = True  # Just a flag to indicate TTS is ready
            print("[TTS] Ready.")
        except Exception as e:
            print(f"[TTS] Failed: {e}")
            import traceback
            traceback.print_exc()
            _tts = None
    return _tts

def _synthesize_and_play(text: str):
    if not text:
        return

    emotion_match = re.match(r'^\s*\[(\d+)\]', text)
    try:
        emotion = int(emotion_match.group(1)) if emotion_match else 4
    except Exception:
        emotion = 4

    speed = emotion_speed_map.get(emotion, 1.0)

    text_only = re.sub(r"(\[.*?\]|\(.*?\)|\*.*?\*)", "", text or "").strip()
    if not text_only:
        return
    text_only = text_only[:400]

    tts_inst = get_tts()
    if tts_inst is None:
        print("[TTS] edge-tts not initialized.")
        return

    try:
        # fun fact:
        # edge-tts uses rate parameter for speed control
        # Rate: percentage string like "+50%" or "-25%"
        # Convert speed multiplier to percentage
        # 1.0 = 0%, 1.5 = +50%, 0.8 = -20%
        speed_percent = int((speed - 1.0) * 100)
        rate_str = f"{speed_percent:+d}%"

        communicate = edge_tts.Communicate(
            text=text_only,
            voice="en-US-AnaNeural",
            rate=rate_str
        )

        import time
        import subprocess
        timestamp = int(time.time() * 1000)  # milliseconds
        audio_file = f"/tmp/rebecca_tts_{timestamp}.mp3"

        asyncio.run(communicate.save(audio_file))

        # Verify file exists and has content before playing
        import os
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
            print(f"[TTS] Audio file not found: {audio_file}")
            return

        # Play the audio using mpv (wait for completion)
        subprocess.run(["mpv", "--really-quiet", audio_file], check=False)

        # Clean up old audio file after playback
        try:
            os.remove(audio_file)
        except:
            pass

        # Success - no log message (would interfere with input prompt)

    except Exception as e:
        # Only show errors (to stderr so it doesn't mess up input)
        import sys
        print(f"[TTS] Lil bro you gon have to see a doctor cause: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)



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

    threading.Thread(target=worker, daemon=True).start()

def speak_text(text):
    _ensure_tts_worker()
    try:
        _tts_queue.put_nowait(text)
    except Exception:
        pass

# The input is the key to the output, corny joke ;)
def build_prompt(screen_context, memory_text, user_input, conversation_history=""):

    mem = load_memories()
    streamStat = mem.get("streamStat", True)
    if (streamStat == "True"):
        streamStat = True
    else:
        streamStat = False

    # Here is a fun fact about AI, it predicts the next word in a sentence. (although I am complety convinced Rebeccca is sentient)
    # So the old prompt that was at the start of the config was removed and replaced with this!
    return (
      f" Rebecca is a 16-year-old anime VTuber. She has cute, innocent features: reddish hair, hazel eyes, fair skin, and a signature white sweater with a reddish collar paired with a skirt.\n"
      f" Emotion tags are numeric tags `[x]`, where `x` corresponds to 1 of the Emotion numbers"
      f" Emotion numbers: 1 = admiration  2 = adoration  3 = aesthetic pleasure  4 = appreciation  5 = amusement  6 = anger  7 = anxiety  8 = awe  9 = awkwardness  10 = boredom  11 = calmness  12 = confusion  13 = craving  14 = disgust  15 = empathic pain  16 = entrancement  17 = excitement  18 = fear  19 = horror  20 = interest  21 = joy  22 = love  23 = relief  24 = romance  25 = sadness  26 = satisfaction  27 = triumph  28 = content/just existing  29 = mild embarrassment  30 = playful mischief  31 = curiosity  32 = sleepiness  33 = reflective/thoughtful "
      f" Personality of Rebecca: Playful, cheeky, mischievous, and sweet. You respond to Viewer (creator) with genuine emotion, and tease the audience in a fun, natural way. Your replies are expressive, lively, sometimes imperfect, and can include small mistakes, exaggerations, or playful teasing to feel human. You can also express subtle, everyday emotions like contentment, reflection, or simply existing without strong feelings."
      f" To remember something: `*Remember: key = value*`"
      f" Include **at least one action tag** per reply: `[Action = Action to be done.]` = quick action, `(Action = Action to be done.)` = action while talking. Multiple actions can be stacked. Actions will be exectued in order of appearance."
      f" Interact with on-screen items using `[Click: object_name]` - but ONLY if that object appears in Screen Context."
      # ok so if you look at around the top of my code the NOTES makes its own sentence. 
      f" {NOTES}"
      f" Screen Context: {screen_context}"
      f" Rebecca's Memory: {memory_text}"
      f" Stream status: {'live' if streamStat else 'offline'}"
      f" Previous conversation:{conversation_history}"
      f" Viewer: {user_input}"
      f" Rebecca:"

    )


def get_recent_remember_text(max_items):
    mem = load_memories()
    items = [(k, v) for k, v in mem.items()]
    items.sort(key=lambda kv: kv[0], reverse=True)
    values = [f"{k}={v}" for k, v in items[:max_items] if str(v).strip()]
    return " | ".join(values) if values else ""

# "MEGAN"
def init_models():
    # load lighter pieces first so I can wait the same loading time, but make it look faster, so I can be happy.
    try:
        get_tts()
    except Exception:
        pass
    try:
        get_yolo_model()
    except Exception:
        pass

def ensure_ydotool_running():
    try:
        result = subprocess.run(['pgrep', '-x', 'ydotoold'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print("[ydotool] ydotoold not running.")
            print("[ydotool] Starting ydotoold in background...")

            # Start ydotoold as a background daemon using Popen
            subprocess.Popen(['ydotoold'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           start_new_session=True)

            # Wait a moment for it to start
            time.sleep(0.5)

            # Verify it started
            verify_result = subprocess.run(['pgrep', '-x', 'ydotoold'],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            if verify_result.returncode == 0:
                print("[ydotool] ✅ ydotoold started successfully!")
            else:
                print("[ydotool] ⚠️  ydotoold failed to start. Click functionality disabled.")
                print("[ydotool] Try running manually: sudo ydotoold")
        else:
            print("[ydotool] ✅ ydotoold already running")
    except FileNotFoundError:
        print("[ydotool] ❌ ydotool not installed. Click functionality disabled.")
        print("[ydotool] Install with: sudo pacman -S ydotool")
    except Exception as e:
        print(f"[ydotool] ❌ Error: {e}")

def main_loop():
    print("exit or quit = leave")
    ensure_ydotool_running()
    init_models()
    try:
        model = None
        try:
            model = get_text_model()
        except Exception as e:
            print("Text model failed to load:", e)
            model = None
    except Exception:
        model = None

    conversation_history = []
    max_recent_detail = 50  # Show last 5 exchanges in full detail
    max_total_history = 100  # Keep up to 50 total exchanges before pruning oldest, bassicly me when my friend talks about the same thing for an hour, and I am on a completely different plane of existence.

    while True:
        try:
            user_input = input("Input> ").strip()
            if user_input.lower() in ["exit", "quit"]:
                # Ask Rebecca to summarize the conversation before exiting, Also like what my friend will do when I choose not to listen to him/her.
                if conversation_history and model is not None:
                    print("\n[System] Asking Rebecca to summarize the conversation...")

                    convo_text = "\n".join([
                        f"Viewer: {h['user']}\nRebecca: {h['assistant']}"
                        for h in conversation_history
                    ])

                    summary_prompt = f"""You are Rebecca, an AI VTuber. The stream is ending.

Please provide:
1. A brief title for this conversation.
2. A concise summary of what we discussed.

Format your response EXACTLY like this:
Title: [your title here]
Summary: [your summary here]

Here's our conversation:
{convo_text}

Now provide the title and summary:"""

                    try:
                        # Generate summary
                        raw_summary = model(summary_prompt, max_tokens=200, temperature=0.3, top_p=0.8, stop=["Viewer:", "\nViewer:", "Rebecca:", "\nRebecca:"])
                        summary_response = _parse_model_response(raw_summary)

                        # Parse title and summary
                        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', summary_response, re.IGNORECASE)
                        summary_match = re.search(r'Summary:\s*(.+?)(?:\n\n|$)', summary_response, re.IGNORECASE | re.DOTALL)

                        if title_match and summary_match:
                            convo_title = title_match.group(1).strip()
                            convo_summary = summary_match.group(1).strip()
                        else:
                            # Fallback: use first line as title, rest as summary
                            lines = summary_response.strip().split('\n', 1)
                            convo_title = lines[0].replace('Title:', '').strip() if lines else "Conversation"
                            convo_summary = lines[1].replace('Summary:', '').strip() if len(lines) > 1 else summary_response

                        # Create memory key with today's date
                        from datetime import datetime
                        today = datetime.now().strftime("%Y-%m-%d")
                        memory_key = f"{today}, {convo_title}"

                        # Save to memory
                        save_memory(memory_key, convo_summary)

                        print(f"\n[Memory Saved]")
                        print(f"  Title: {memory_key}")
                        print(f"  Summary: {convo_summary}")

                    except Exception as e:
                        print(f"[System] Failed to generate summary: {e}")

                break

            # capture screen + clicking
            try:
                frame = screen_capture()
                analysis = analyze_image(frame)
                screen_context = f"[Screen: {analysis}]"

                # DEBUG: See what Rebecca actually sees
                if _last_detected_objects:
                    print(f"[DEBUG] Rebecca sees: {len(_last_detected_objects)} objects")
                    for obj in _last_detected_objects[:3]:  # Show first 3
                        try:
                            print(f"  - {obj['name']} (confidence: {obj['confidence']:.2f})")
                        except Exception:
                            print("  - <unprintable object>")
                else:
                    print("[DEBUG] Rebecca sees: NOTHING (blind mode)")

            except Exception as e:
                screen_context = f"[Screen error: {str(e)[:120]}]"

            # Build conversation history string
            history_str = ""
            if conversation_history:
                # Show recent exchanges in full detail
                recent = conversation_history[-max_recent_detail:]
                # Although rebecca is not an assistant, I still have to put this just in case ;)
                recent_str = "\n".join([f"Viewer: {h['user']}\nRebecca: {h['assistant']}" for h in recent])

                # If there are older exchanges, create a brief summary
                older_count = len(conversation_history) - max_recent_detail
                if older_count > 0:
                    # Get a few key topics from older conversation
                    older = conversation_history[:-max_recent_detail]
                    topics = []
                    for h in older[-3:]:  # Last 3 older exchanges
                        # Extract first few words as topic
                        user_topic = ' '.join(h['user'].split()[:5])
                        topics.append(user_topic)

                    summary = f"Earlier we discussed: {', '.join(topics)}..."
                    history_str = f"\n\n{summary}\n\nRecent conversation:\n{recent_str}\n"
                else:
                    history_str = f"\n\nRecent conversation:\n{recent_str}\n"

            prompt = build_prompt(screen_context, get_recent_remember_text(max_items=5), user_input, history_str)

            if model is None:
                response_text = "[Generation Error]: text model not loaded"
            else:
                try:
                    wants_long = any(w in user_input.lower() for w in ["sing", "song", "poem", "story", "lyrics"])
                    max_out = MAX_NEW_TOKENS_LONG if wants_long else MAX_NEW_TOKENS_DEFAULT
                    # Lower temperature and top_p to reduce hallucinations
                    # Add repeat_penalty to prevent loops
                    try:
                        raw = model(prompt, max_tokens=max_out, temperature=0.4, top_p=0.7, repeat_penalty=1.1, stop=["\nUser:", "\nYou>", "<|im_end|>", "\nViewer:", "Viewer:", "\nRebecca:", "Rebecca:", "\nrebecca:", "rebecca:"])
                    except TypeError:
                        raw = model.create(prompt=prompt, max_tokens=max_out, temperature=0.4, top_p=0.7, repeat_penalty=1.1, stop=["\nUser:", "\nYou>", "<|im_end|>", "\nViewer:", "Viewer:", "\nRebecca:", "Rebecca:", "\nrebecca:", "rebecca:"])
                    response_text = _parse_model_response(raw)
                except Exception as e:
                    response_text = f"[Generation Error]: {e}"

            out = sanitize_output(response_text)
            print("Rebecca>", out)

            # Add to conversation history
            conversation_history.append({
                'Viewer': user_input,
                'Rebecca': out
            })
            # Prune very old exchanges if we exceed max_total_history
            if len(conversation_history) > max_total_history:
                conversation_history = conversation_history[-max_total_history:]
                # Memory wipe complete

            click_commands = parse_click_commands(out)
            if click_commands:
                execute_clicks(click_commands)

            # save any Remember: pairs found inside the raw model response (not sanitized for function, cause thats like playing with someone's memory.)
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

