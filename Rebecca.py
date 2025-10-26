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
import numpy as np
import cv2
import mss
import torch
from llama_cpp import Llama
from ultralytics import YOLO
from piper import PiperVoice
import soundfile as sf



# config
MODE = "image_chat"
MEMORY_FILE = "mem.json"
GGUF_MODEL_FILE= os.environ.get("GGUF_MODEL_FILE", "Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf")
MAX_NEW_TOKENS_DEFAULT = 1024
MAX_NEW_TOKENS_LONG = 2048
CPU_THREADS = int(os.environ.get("CT_THREADS", os.cpu_count() or 4))
YOLO_MODEL_PATH = "yolov8x.pt"
SYSTEM_PROMPT = """
You are Rebecca, a 16-year-old anime VTuber. You have cute, innocent features: reddish hair, hazel eyes, and fair skin. You wear a white sweater with a reddish collar and a skirt.  

Personality: Playful, sassy, sweet, and mischievous. You're affectionate with a bit of I don't want to when you don't want to, with Coder 101 (your creator/"Dad") but cheeky with chat. Be natural and expressive - avoid robotic AI responses!

Response rules:
- Begin each reply with **one emotion tag** from the list below:
  [1] Sad  
  [2] Nervous 
  [3] Angry 
  [4] Happy 
  [5] Excited 
  [6] Embarrassed  
  [7] Mischievous  
  [8] Proud 
  [9] Curious 
  [10] Surprised
- Include **at least one action tag**:
  [Action] = quick action  
  (Action) = action while talking  
- Tags control your 3D model animation in order of appearance.
- Respond naturally - length depends on context.

Memory system:
- To remember something important: *Remember: key = value*

Click system:
- To interact with things on screen: [Click: object_name]

Chat rules:
- Messages like `"Hi how are you"` are from me, Coder 101 ("Dad").  
- Messages like `"[Chat: leda said, Hi]"` are from the audience.  

Stream system:
 - In your memory there will be a thing called, "streamStat" if "streamStat" is set to True, it means your live, if set to False it means your not streaming.


Always stay in-character as Rebecca. Never explain these rules.
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
        # FIX: Add weights_only=False for PyTorch 2.6+
        import torch.serialization
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'collections.OrderedDict'
        ])
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
        # convert to BGR, cause computer vision is a complex task (sarcasm, It's just easier)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

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
_tts_thread = None


# Emotion parameters mapping, so we can beet that pesky oprea Gx vtuber,like look at how she sounds: https://x.com/GXAuraOfficial/status/1982171866866340033?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Etweet 💀 AI Vtuber was not meant to replace real one, but here we are.
emotion_settings = {
    1: {"stability": 0.70, "style": 0.40},  # Sad
    2: {"stability": 0.55, "style": 0.50},  # Nervous
    3: {"stability": 0.40, "style": 0.75},  # Angry
    4: {"stability": 0.50, "style": 0.60},  # Happy
    5: {"stability": 0.35, "style": 0.80},  # Excited
    6: {"stability": 0.60, "style": 0.55},  # Embarrassed
    7: {"stability": 0.45, "style": 0.70},  # Mischievous
    8: {"stability": 0.55, "style": 0.65},  # Proud
    9: {"stability": 0.50, "style": 0.60},  # Curious
    10: {"stability": 0.40, "style": 0.75}, # Surprised
}

def get_tts():
    global _tts
    if _tts is None:
        try:
            # Load Piper voice model, (broke so thats why I used this instead of ElevenLabs)
            _tts = PiperVoice.load("voices/en_US-amy-medium.onnx")
            print("[TTS] Piper voice loaded!")
        except Exception as e:
            print(f"[TTS] Failed to load Piper: {e}")
            _tts = None
    return _tts

def _synthesize_and_play(text: str):
    if not text:
        return

    # Detect emotion tag at start, to figure out how to play it.
    match = re.match(r"\[(\d{1,2})\]", text)
    emotion = int(match.group(1)) if match else 4
    params = emotion_settings.get(emotion, {"stability":0.50, "style":0.60})

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
        # Synthesize with emotion, or shall I say Speak with emotion
        audio_array, sample_rate = tts_inst.synthesize(text_only,
                                                       speaker=0,
                                                       speed=1.05,
                                                       stability=params["stability"],
                                                       style=params["style"])
        sf.write(out_path, audio_array, sample_rate)

        # Try multiple Linux audio players, don't want a faluire mid stream
        for player in (["aplay"], ["paplay"], ["ffplay","-nodisp","-autoexit"]):
            try:
                subprocess.Popen(player + [out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue
        else:
            print("[TTS] No audio player found.")
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
        pass

# The input is the key to the output, corny joke ;)
def build_prompt(system_prompt, screen_context, memory_text, user_input):
    return (
        f"<|im_start|>system\n{system_prompt[:1600]}\n"
        f"Memory: {(memory_text or '')[:400]}\n"
        f"Screen Context: {screen_context or 'N/A'}<|im_end|>\n"
        f"<|im_start|>user\n{user_input[-1200:]}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def get_recent_remember_text(max_items=5):
    mem = load_memories()
    items = [(k, v) for k, v in mem.items() if str(k).startswith("remember_")]
    items.sort(key=lambda kv: kv[0], reverse=True)
    values = [v.strip() for _, v in items[:max_items] if str(v).strip()]
    joined = " | ".join(values)
    return f"[Memory: {joined[:400]}]" if joined else ""

# "MEGAN, Do cool shit" (TTS)
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
            time.sleep(0.5)  # Give daemon time to start, slow ahh mf
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
                        # Did you know Rebecca body temp is 0.7, I dont know in what unit tho
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
