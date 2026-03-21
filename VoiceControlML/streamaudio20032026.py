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
    "Base",              # 1
    "Cycle",             # 2
    "Arm",               # 3
    "Down",              # 4
    "Drop",              # 5
    "Forward",           # 6
    "Left",              # 7
    "Reverse",           # 8
    "Right",             # 9
    "Spin",              # 10
    "Stop",              # 11
    "Up"                 # 12
]

#sensitivities, lower = more sensitive
CLASS_THRESHOLDS = {
    "Background Noise": 50,
    "Cycle": 95,
    "Arm": 95,
    "Down": 95,
    "Drop": 95,
    "Forward": 95,
    "Base": 95,
    "Left": 95,
    "Reverse": 95,
    "Right": 95,
    "Spin": 95,
    "Stop": 95,
    "Up": 95
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

#confirmation logic
drop_counter = 0
prev_drop_state = False
drop_last_time = 0

land_counter = 0
prev_land_state = False
land_last_time = 0

arm_counter = 0
prev_arm_state = False
arm_last_time = 0

raw_max_confidence = 0.185

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

        if last_command == "Left":
            if spin_toggle:
                display_command = "Yaw Left"
            else:
                display_command = "Roll Left"
        elif last_command == "Right":
            if spin_toggle:
                display_command = "Yaw Right"
            else:
                display_command = "Roll Right"

        if last_command == "Drop" and not prev_drop_state:
            drop_counter += 1
            drop_last_time = time.time()
        
        prev_drop_state = (last_command == "Drop")
        
        if time.time() - drop_last_time > 3.0:
            drop_counter = 0

        if last_command == "Base" and not prev_land_state:
            land_counter += 1
            land_last_time = time.time()

        prev_land_state = (last_command == "Base")

        if time.time() - land_last_time > 3.0:
            land_counter = 0

        if last_command == "Arm" and not prev_arm_state:
            arm_counter += 1
            arm_last_time = time.time()

        prev_arm_state = (last_command == "Arm")

        if time.time() - arm_last_time > 3.0:
            arm_counter = 0

        if last_command == "Drop":
            if drop_counter >= 2:
                display_command = "Confirm Drop"
            elif drop_counter == 1:
                display_command = "Request Drop"
        elif last_command == "Base":
            if land_counter >= 2:
                display_command = "Confirm Base"
            elif land_counter == 1:
                display_command = "Request Base"
        elif last_command == "Arm":
            if arm_counter >= 2:
                display_command = "Confirm Arm"
            elif arm_counter == 1:
                display_command = "Request Arm"

        if display_command != prev_command:

            if display_command == "Cycle":
                pass
            elif display_command == "Down":
                pass
            elif display_command == "Confirm Drop":
                pass
            elif display_command == "Request Drop":
                pass
            elif display_command == "Forward":
                pass
            elif display_command == "Roll Left":
                pass
            elif display_command == "Roll Right":
                pass
            elif display_command == "Reverse":
                pass
            elif display_command == "Stop":
                pass
            elif display_command == "Up":
                pass
            elif display_command == "Yaw Left":
                pass
            elif display_command == "Yaw Right":
                pass
            elif display_command == "Confirm Base":
                pass
            elif display_command == "Request Base":
                pass
            elif display_command == "Confirm Arm":
                pass
            elif display_command == "Request Arm":
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
        line = " ".join(list_to_print) + f" | Display: \033[96m{display_command}\033[0m" + f" | Yaw?: \033[96m{spin_toggle}\033[0m"

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
