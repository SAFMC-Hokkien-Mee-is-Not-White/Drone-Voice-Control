import sounddevice as sd
import numpy as np
import queue
import whisper
from NLP.parser import parse_command
from NLP.commands import execute
import time

q = queue.Queue()
model = whisper.load_model("small")  # use 'base' for lower latency
samplerate = 16000
buffer_duration = 2  # seconds
buffer = np.zeros((0,), dtype=np.float32)

# Audio callback
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def main(vehicle):
    global buffer
    with sd.InputStream(channels=1, samplerate=samplerate, callback=audio_callback):
        print("[INFO] Listening for commands...")
        while True:
            # collect available chunks
            while not q.empty():
                data = q.get()
                data = data.flatten().astype(np.float32)  # sounddevice default is float32
                buffer = np.concatenate((buffer, data))

                # Keep only last N seconds
                max_samples = buffer_duration * samplerate
                if len(buffer) > max_samples:
                    buffer = buffer[-max_samples:]

            # Only transcribe if we have at least 0.5s of audio
            if len(buffer) > samplerate // 2:
                result = model.transcribe(buffer, fp16=False)
                text = result["text"].strip()
                if text:
                    print(f"[STT] Recognized: {text}")
                    command, distance = parse_command(text)
                    execute(vehicle, command, distance)
                    buffer = np.zeros((0,), dtype=np.float32)  # reset buffer after command

            time.sleep(0.1)