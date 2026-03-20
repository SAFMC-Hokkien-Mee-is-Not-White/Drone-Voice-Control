
#This uses ML from https://teachablemachine.withgoogle.com/train


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import sounddevice as sd
from collections import deque
import sys
import time

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Settings ---
SAMPLE_RATE = 44100
WINDOW_DURATION = 1.0
STEP_DURATION = 0.1
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE = int(SAMPLE_RATE * STEP_DURATION)

buffer = deque(maxlen=WINDOW_SIZE)
for _ in range(WINDOW_SIZE):
    buffer.append(0.0)

#from the labels.txt file
CLASS_NAMES = [
    "Background Noise",  # 0
    "Cycle",             # 1
    "Down",              # 2
    "Drop",              # 3
    "Forward",           # 4
    "Left",              # 5
    "Reverse",           # 6
    "Right",             # 7
    "Spin",              # 8
    "Stop",              # 9
    "Up"                 # 10
]

#sensitivities, lower = more sensitive
CLASS_THRESHOLDS = {
    "Background Noise": 50,
    "Cycle": 21,
    "Down": 20,
    "Drop": 20,
    "Forward": 21,
    "Left": 21,
    "Reverse": 21,
    "Right": 21,
    "Spin": 21,
    "Stop": 21,
    "Up": 21
}

general_sensitivity = 1 #lower = more sensitive. Best keep it at 1 for stability

for i in CLASS_THRESHOLDS:
    CLASS_THRESHOLDS[i] = CLASS_THRESHOLDS[i]*general_sensitivity

COMMAND_HOLD_TIME = 0.5
last_command = "None"
last_command_time = 0

# --- Spin toggle state (RESTORED) ---
spin_toggle = False
prev_raw_cmd = "None"

drop_counter = 0
prev_drop_state = False

# Reserve enough lines for multi-line probability display + 2 command lines
NUM_LINES = 3 + 2  # Back to original
print("\n" * NUM_LINES)

# --- Functions ---
def preprocess_audio(audio):
    audio = np.array(audio, dtype=np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    audio = audio.flatten()
    audio = audio[:input_details[0]['shape'][1]]
    if len(audio) < input_details[0]['shape'][1]:
        audio = np.pad(audio, (0, input_details[0]['shape'][1] - len(audio)))
    return np.expand_dims(audio, axis=0).astype(np.float32)

def predict(audio_chunk):
    input_data = preprocess_audio(audio_chunk)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return tf.nn.softmax(output_data[0]).numpy()

def format_probs_live(probs, chunk_size=4):
    """Split probabilities into multiple lines and highlight max in yellow"""
    max_idx = np.argmax(probs)
    lines = []
    for i in range(0, len(probs), chunk_size):
        chunk = probs[i:i+chunk_size]
        names = CLASS_NAMES[i:i+chunk_size]
        line_parts = []
        for j, (name, p) in enumerate(zip(names, chunk)):
            perc = f"{p*100 if not np.isnan(p*100) else 0:5.1f}%"
            if i+j == max_idx:
                line_parts.append(f"\033[93m{name[:5]}:{perc}\033[0m")
            else:
                line_parts.append(f"{name[:5]}:{perc}")
        lines.append(" | ".join(line_parts))
    return lines, max_idx

def audio_callback_in(indata, frames, time_info, status):
    global last_command, last_command_time, drop_counter, prev_drop_state
    global spin_toggle, prev_raw_cmd

    buffer.extend(indata[:, 0])

    if len(buffer) == WINDOW_SIZE:
        probs = predict(list(buffer))
        max_idx = np.argmax(probs)
        max_conf = probs[max_idx]*100
        cmd_name = CLASS_NAMES[max_idx]
        threshold = CLASS_THRESHOLDS.get(cmd_name, 21)
        
        # Only consider non-background commands that meet threshold
        if max_idx != 0 and max_conf >= threshold:
            raw_cmd = cmd_name
        else:
            raw_cmd = "None"

        # --- Spin toggle detection ---
        if raw_cmd == "Spin" and prev_raw_cmd != "Spin":
            spin_toggle = not spin_toggle
        prev_raw_cmd = raw_cmd

        now = time.time()
        if raw_cmd != "None":
            last_command = raw_cmd
            last_command_time = now
        elif now - last_command_time > COMMAND_HOLD_TIME:
            last_command = "None"

        




        #PUT IN YOUR COMMANDS DOWN HERE \/
        
        display_command = last_command
        
        if last_command == "Cycle":
            display_command = "CYCLE"
            # Add your cycle command here
            
        elif last_command == "Down":
            display_command = "DOWN"
            # Add your down command here
            
        elif last_command == "Drop":
            # Drop confirmation logic
            if last_command == "Drop" and not prev_drop_state:
                drop_counter += 1
                last_command_time = now
            
            if time.time() - last_command_time > 3.0:
                drop_counter = 0
                
            if drop_counter >= 2:
                display_command = "CONFIRMED DROP"
                # Add your drop command here
                drop_counter = 0  # Reset after confirmed
            else:
                display_command = "REQUEST DROP"
                
        elif last_command == "Forward":
            display_command = "FORWARD"
            # Add your forward command here
            
        elif last_command == "Left":
            if spin_toggle:
                display_command = "YAW LEFT"
                # Add your yaw left command here
            else:
                display_command = "ROLL LEFT"
                # Add your roll left command here
            
        elif last_command == "Reverse":
            display_command = "REVERSE"
            # Add your reverse command here
            
        elif last_command == "Right":
            # --- Spin toggle affects Right command (RESTORED) ---
            if spin_toggle:
                display_command = "YAW RIGHT"
                # Add your yaw right command here
            else:
                display_command = "ROLL RIGHT"
                # Add your roll right command here
            
        elif last_command == "Spin":
            display_command = "SPIN"
            # Add your spin command here
            
        elif last_command == "Stop":
            display_command = "STOP"
            # Add your stop command here
            
        elif last_command == "Up":
            display_command = "UP"
            # Add your up command here
            
        # =====================================================





        prev_drop_state = (last_command == "Drop")

        # --- Live UI update ---
        prob_lines, _ = format_probs_live(probs)
        sys.stdout.write("\033[F" * NUM_LINES)
        for line in prob_lines:
            sys.stdout.write("\033[K"); print(line)

        # Show spin toggle status in raw command line
        sys.stdout.write("\033[K"); print(f"Raw Command: {raw_cmd} | Spin Mode: {'YAW' if spin_toggle else 'ROLL'}")
        sys.stdout.write("\033[K"); print(f"\033[92mCommand: {display_command}\033[0m")
        sys.stdout.flush()


# --- Main loop ---
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=STEP_SIZE, callback=audio_callback_in):
    try:    
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped")


#Forwards, Reverse, Up, Down, Left, Right   :       Movement of drone
#Spin                                       :       Switches between Yaw mode, and Roll mode. (e.g. Yaw left, Roll left)
#Cycle, Drop                                :       Spins the drum, and drops the payload respectively
#Stop                                       :       Stops all movement
