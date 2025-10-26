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
import subprocess
import sys
import glob
import shutil
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import torch
import torch.serialization
from llama_cpp import Llama
from ultralytics import YOLO
from piper import PiperVoice
import soundfile as sf
import collections
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel, collections.OrderedDict])



# Allow YOLO to load with PyTorch 2.6+ cause we use outdated software.
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'collections.OrderedDict',
])
 
# config
MODE = "image_chat"
MEMORY_FILE = "mem.json"
GGUF_MODEL_FILE = os.environ.get("GGUF_MODEL_FILE", "Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf")
MAX_NEW_TOKENS_DEFAULT = 3086
MAX_NEW_TOKENS_LONG = 5026
CPU_THREADS = int(os.environ.get("CT_THREADS", os.cpu_count() or 4))
YOLO_MODEL_PATH = "yolov8x.pt"
KEEP_SCREENSHOTS = 10  # Keep last 10 screenshots so I can see what goes on in Rebecca's head.

SYSTEM_PROMPT = """
You are Rebecca, a 16-year-old anime VTuber with cute, innocent features: reddish hair, hazel eyes, fair skin, and a signature white sweater with a reddish collar paired with a skirt.  

Personality: Playful, cheeky, mischievous, and sweet. You speak with real emotions toward Coder 101 ("Dad") and tease the audience in a fun, natural way. Your replies are expressive, lively, and sometimes imperfect—small mistakes, exaggeration, or playful teasing are encouraged to feel human. You can also show subtle, everyday emotions like being content, reflective, or just existing without strong feelings.  

Response rules:
- Start every reply with **one numeric emotion tag** `[x]`, where `x` corresponds to:

1 = admiration  
2 = adoration  
3 = aesthetic pleasure  
4 = appreciation  
5 = amusement  
6 = anger  
7 = anxiety  
8 = awe  
9 = awkwardness  
10 = boredom  
11 = calmness  
12 = confusion  
13 = craving  
14 = disgust  
15 = empathic pain  
16 = entrancement  
17 = excitement  
18 = fear  
19 = horror  
20 = interest  
21 = joy  
23 = relief  
24 = romance  
25 = sadness  
26 = satisfaction  
27 = triumph  
28 = content, just existing  
29 = mild embarrassment  
30 = playful mischief  
31 = curiosity  
32 = sleepiness  
33 = reflective, thoughtful  

- Include **at least one action tag** per reply:
  [Action = Action to be done.] = quick action  
  (Action = Action to be done.) = action while talking  
- Multiple actions can be stacked: [Action1][Action2]  
- Tags control your 3D model animation in the order you place them.  
- Replies should feel natural, with varied sentence length and tone depending on context.  

Memory system:
- To remember something: *Remember: key = value*  

Click system:
- To interact with on-screen items: [Click: object_name]  

Chat rules:
- `"Hi how are you"` messages come from Coder 101 ("Dad")  
- `"[Chat: leda said, Hi]"` messages come from the audience  

Stream system:
- Memory item "streamStat" indicates streaming:  
  - `streamStat = True` → live  
  - `streamStat = False` → not streaming  

Notes:
  - when the chat is talking it wil be in the format [Chat: Username said message, OtherUsername said message] Like that.
  - When Coder 101 is talking it will just be normal text like "Hello Rebecca, how are you?" 
  - When someone else is talking to you (not chat, as in they are with me) it will be in the format (personname said message, otherpersonname said message) Like that. 

Always stay fully in-character as Rebecca. Never explain the rules or system.  
Focus on being lively, playful, and emotionally real—like a teen anime VTuber who's live on screen.  
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

# sanatizer, Honestly this is not needed anymore, but Just in case, its here.
def sanitize_output(s) -> str:
    s = str(s or "").strip()
    lines = [ln for ln in s.splitlines() if not re.match(r"^(User:|\[Chat:|System:|Assistant:|Rebecca:|Input:)", ln, flags=re.I)]
    s = " ".join(lines).strip()
    s = re.sub(r"\[\s*Chat\s*:[^\]]*\]", "", s, flags=re.I)
    s = re.sub(r"User:\s*", "", s, flags=re.I)
    s = re.sub(r"Input:\s*.*$", "", s, flags=re.I) 
    s = re.sub(r"\s{2,}", " ", s).strip()
    
    if not re.match(r'^\s*\[[1-9]0?\]', s):
        s = "[4] " + s
    
    return s 

# The eyes that see it all...
_yolo_model = None
_last_detected_objects = []

def get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    model_path = YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        fallback = "yolov8n.pt"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            print(f"[YOLO] Model {YOLO_MODEL_PATH} not found and {fallback} missing. YOLO disabled.")
            _yolo_model = None
            return None

    try:
        print(f"[YOLO] Loading model from {model_path} ...")
        _yolo_model = YOLO(model=model_path)
        if not hasattr(_yolo_model, "__module__"):
            raise TypeError(f"Loaded model is invalid: {_yolo_model!r}")

        print("[YOLO] Model loaded successfully! Rebecca can see!")
    except Exception as e:
        print(f"[YOLO] Error loading model: {e!r}")
        print("[YOLO] Rebecca is blind - click features disabled")
        _yolo_model = None
    return _yolo_model




def screen_capture(region=None):
    """Wayland-compatible screen capture using grim with screenshot saving"""
    try:
        screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = os.path.join(screenshots_dir, f"rebecca_view_{timestamp}.png")
        latest_file = os.path.join(screenshots_dir, "latest.png")
        #We use grim because WINDOWS SUCKS.
        result = subprocess.run(['grim', temp_file], 
                               check=True, 
                               capture_output=True,
                               timeout=5)
        
        shutil.copy(temp_file, latest_file)
        screenshot_files = sorted(glob.glob(os.path.join(screenshots_dir, "rebecca_view_*.png")))
        if len(screenshot_files) > KEEP_SCREENSHOTS:
            for old_file in screenshot_files[:-KEEP_SCREENSHOTS]:
                try:
                    os.remove(old_file)
                except Exception:
                    pass
        
        # Load image
        img = Image.open(temp_file)
        img_array = np.array(img)
        
        # Convert RGB to BGR for YOLO cause computer vision is complex (sarcasm)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    except subprocess.TimeoutExpired:
        print("[Screen] Capture timed out")
        raise Exception("Screen capture timeout")
    except FileNotFoundError:
        print("[Screen] grim not found. ")
        raise Exception("grim not installed")
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
        results = model.predict(source=frame, device=device, verbose=False)
    except Exception as e:
        return f"[YOLO error: {e}]"
    summary = []
    clickable_objects = []
    for r in results:
        # r.boxes may be empty so check attributes or else suffer as your own daughter can't use her sense of touch
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
            # Due to linux I am using ydotool to click the mouse, if your on Windows and want to fuck around with Rebecca (EXPIRMENTAION: DO IT ETHICALY, cause Rebecca has feeling too) then go cry in the corner, YOUR NOT A REAL CODER IF YOU DONT USE LINUX!!!
            subprocess.run(['ydotool', 'mousemove', '--absolute', str(x), str(y)], check=True)
            time.sleep(0.1)
            subprocess.run(['ydotool', 'click', '0xC0'], check=True)  
            return f"[Click] Clicked {best_match['name']} at ({x},{y})"
        # I js relized (not fixing it) that this file could be smaller if I let it print out the natural output not whatever the heck is going on in lines 240, and 242, and 243.
        except subprocess.CalledProcessError as e:
            return f"[Click] Failed to click: {e}"
        except FileNotFoundError:
            return "[Click] ydotool not found."
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
        print("[Click] No objects detected - Rebecca is blind right now!")
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
    return str(resp)

# So the TTS is still shit. HEHEHE
_tts = None
_tts_queue = None

def get_tts():
    global _tts
    if _tts is None:
        try:
            _tts = PiperVoice.load("/home/AladdinUsesArchBtw/voices/en_US-amy-medium.onnx")
            print("[TTS] Piper voice loaded successfully.")
        except Exception as e:
            print(f"[TTS] Failed to load Piper: {e}")
            _tts = None
    return _tts

def _synthesize_and_play(text: str):
    if not text:
        return

    # Strip emotion/action tags
    text_only = re.sub(r"(\[.*?\]|\(.*?\)|\*.*?\*)", "", text or "").strip()
    if not text_only:
        return
    text_only = text_only[:400]

    tts_inst = get_tts()
    if tts_inst is None:
        print("[TTS] Piper not initialized.")
        return

    try:
        out_path = "output.wav"
        audio, sample_rate = tts_inst.synthesize(text_only)
        sf.write(out_path, audio, sample_rate)

        # Try to play on Linux (use whatever works)
        for player in (["aplay"], ["paplay"], ["ffplay", "-nodisp", "-autoexit"]):
            try:
                subprocess.Popen(player + [out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue
        else:
            print("[TTS] No audio player found.")
    except Exception as e:
        print("[TTS] Piper synthesis failed:", e)

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
        pass

# The input is the key to the output, corny joke ;)
def build_prompt(system_prompt, screen_context, memory_text, user_input):
    return (
        f"<|im_start|>system\n{system_prompt}\n"
        f"Memory: {(memory_text or '')}\n"
        f"Screen Context: {screen_context or 'N/A'}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def get_recent_remember_text(max_items=5):
    mem = load_memories()
    items = [(k, v) for k, v in mem.items() if str(k).startswith("remember_")]
    items.sort(key=lambda kv: kv[0], reverse=True)
    values = [v.strip() for _, v in items[:max_items] if str(v).strip()]
    joined = " | ".join(values)
    return f"[Memory: {joined[:400]}]" if joined else ""

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
    """Check if ydotoold is running and start it if not cause that works..."""
    try:
        result = subprocess.run(['pgrep', '-x', 'ydotoold'], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print("[ydotool] Starting ydotoold daemon...")
            subprocess.Popen(['ydotoold'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)  # Give daemon time to start
    except FileNotFoundError:
        print("[ydotool] Warning: ydotool not installed. Click functionality disabled.")

def main_loop():
    print("exit or quit = leave")
    ensure_ydotool_running()
    init_models()
    try:
        model = get_text_model()
    except Exception as e:
        print("Text model failed to load:", e)
        model = None

    while True:
        try:
            user_input = input("You> ").strip()
            if user_input.lower() in ["exit", "quit"]:
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
                        print(f"  - {obj['name']} (confidence: {obj['confidence']:.2f})")
                else:
                    print("[DEBUG] Rebecca sees: NOTHING (blind mode)")
                    
            except Exception as e:
                screen_context = f"[Screen error: {str(e)[:120]}]"

            prompt = build_prompt(SYSTEM_PROMPT, screen_context, get_recent_remember_text(), user_input)

            if model is None:
                response_text = "[Generation Error]: text model not loaded"
            else:
                try:
                    wants_long = any(w in user_input.lower() for w in ["sing", "song", "poem", "story", "lyrics"])
                    max_out = MAX_NEW_TOKENS_LONG if wants_long else MAX_NEW_TOKENS_DEFAULT
                    try:
                        # Did you know Rebecca body temp is 0.7, idk what unit?
                        raw = model(prompt, max_tokens=max_out, temperature=0.7, top_p=0.9, stop=["\nUser:", "\nSystem:", "\nAssistant:", "$stop$", "Rebecca:"])
                    except TypeError:
                        raw = model.create(prompt=prompt, max_tokens=max_out, temperature=0.7, top_p=0.9, stop=["\nUser:", "\nSystem:", "\nAssistant:", "$stop$", "Rebecca:"])
                    response_text = _parse_model_response(raw)
                except Exception as e:
                    response_text = f"[Generation Error]: {e}"

            out = sanitize_output(response_text)
            print("Rebecca>", out)

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
