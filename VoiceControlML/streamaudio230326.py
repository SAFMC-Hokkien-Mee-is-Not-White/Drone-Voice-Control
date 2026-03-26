'''
This uses ML from https://teachablemachine.withgoogle.com/train
'''

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

prev_command = "None"

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
    "Blue",              # 1
    "Green",             # 2
    "Left",              # 3
    "Orange",            # 4
    "Right",             # 5
    "Yellow"             # 6
]

#lower = more sensitve
CLASS_THRESHOLDS = {
    "Background Noise": 50,
    "Blue": 95,
    "Green": 95,
    "Left": 95,
    "Orange": 95,
    "Right": 95,
    "Yellow": 95
}

general_sensitivity = 1 #lower = more sensitive. Best keep it at 1 for stability

for i in CLASS_THRESHOLDS:
    CLASS_THRESHOLDS[i] = CLASS_THRESHOLDS[i]*general_sensitivity

COMMAND_HOLD_TIME = 0.5
last_command = "None"
last_command_time = 0

# toggle logic
spin_toggle = False
prev_raw_cmd = "None"

#to set this:
#   1) put it to 0, this outputs the raw percentage for each class
#   2) set it to that percentage (in absolute value, not %)
raw_max_confidence = 0.315 

NUM_LINES = 3
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

def scale_confidence(conf, max_conf=raw_max_confidence): #number goes down wheres theres more classes
    """Scale confidence from 0-max_conf to 0-100"""
    return min(100, (conf / max_conf) * 100)

def predict(audio_chunk):
    input_data = preprocess_audio(audio_chunk)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return tf.nn.softmax(output_data[0]).numpy()

def audio_callback_in(indata, frames, time_info, status):
    global last_command, last_command_time, drop_counter, prev_drop_state, drop_last_time
    global land_counter, prev_land_state, land_last_time
    global arm_counter, prev_arm_state, arm_last_time
    global spin_toggle, prev_raw_cmd

    buffer.extend(indata[:, 0])

    if len(buffer) == WINDOW_SIZE:
        probs = predict(list(buffer))
        max_idx = np.argmax(probs)
        max_conf = scale_confidence(probs[max_idx]*100, raw_max_confidence*100) #number goes down wheres theres more classes
        cmd_name = CLASS_NAMES[max_idx]
        threshold = CLASS_THRESHOLDS.get(cmd_name, 21)
        
        if max_idx != 0 and max_conf >= threshold:
            raw_cmd = cmd_name
        else:
            raw_cmd = "None"

        if raw_cmd == "Spin" and prev_raw_cmd != "Spin":
            spin_toggle = not spin_toggle
        prev_raw_cmd = raw_cmd

        now = time.time()
        if raw_cmd != "None":
            last_command = raw_cmd
            last_command_time = now
        elif now - last_command_time > COMMAND_HOLD_TIME:
            last_command = "None"

        global prev_command

        display_command = last_command

        # --- Background Noise explicit handling ---
        if last_command == "Background Noise" or last_command == "None":
            display_command = "Idle"

        # --- Direction logic ---
        elif last_command == "Left":
            display_command = "Left"

        elif last_command == "Right":
            display_command = "Right"

        # --- Pass-through commands ---
        elif last_command == "Blue":
            display_command = "Blue"

        elif last_command == "Green":
            display_command = "Green"

        elif last_command == "Orange":
            display_command = "Orange"

        elif last_command == "Yellow":
            display_command = "Yellow"



        if display_command != prev_command:

            if display_command == "Blue":
                pass
            elif display_command == "Green":
                pass
            elif display_command == "Left":
                pass
            elif display_command == "Right":
                pass
            elif display_command == "Orange":
                pass
            elif display_command == "Yellow":
                pass

            else:
                # Background / Idle case
                pass

        prev_command = display_command

        # --- Safe max index for highlighting ---
        safe_probs = np.nan_to_num(probs, nan=-np.inf)
        max_idx = int(np.argmax(safe_probs))

        # --- Build the probability display ---
        list_to_print = []
        for i in range(len(CLASS_NAMES)):
            value_str = f"{CLASS_NAMES[i]}:{probs[i]*100/raw_max_confidence:.1f}%"
            if i == max_idx:
                # Highlight the class with highest probability in yellow
                value_str = f"\033[93m{value_str}\033[0m"
            list_to_print.append(value_str)

        # --- Add the display_command at the end, highlighted cyan for visibility ---
        line = " ".join(list_to_print) + f" | Display: \033[96m{display_command}\033[0m"

        # --- Print in place, fixed-width to prevent scrolling ---
        print("\r" + line.ljust(400), end="", flush=True)


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
#Base, Arm                                  :       Lands Drone, arms/disarms drone