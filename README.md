Voice control interface for a custom built drone to drop payload via an electromagnet. A PixHawk 4 is being used to communicate with the drone via telemetry.

Current folder architecture
```
SAFMC_Hokkien-Mee-is-Not-White
  ├── main.py
  ├── config.py
  └── keyboard_control.py
  ├── Drone/
      ├── connection.py
      └── controller.py
  ├── NLP/
      ├── commands.py
      └── parser.py
  ├── STT/                                # Not used for now
      └── whisper_stt.py
  ├── Utility/
      └── audio.py                        # Not used for now
  └── VoiceControlML/
      ├── streamaudio20032026.py
      ├── labels.txt
      └── soundclassifier_with_metadata.tflite
```
<h3>keyboard_control.py</h3>
Current code can be manually controlled without the parsing or processing of voice input, via ```keyboard_control.py```. 

To test via ```keyboard_control.py```: (Note run in venv on Rasberry Pi 5)
```python3 keyboard_control.py```
After connection of vehicle, following commands are available:
 1. ```Arm``` - arms drone and takes off drone to altitude according to speed set in QGroundControl according to parameters ```MIS_TAKEOFF_ALT``` and ```MPC_TKO_SPEED``` respectively (remember updaate the variables in controller and config files).
 2. ```Forward x``` - where x is the number of metres the drone should move forward
 3. ```Backward x``` - where x is the number of metres the drone should move backward
 4. ```Left x``` - where x is the number of metres the drone should move left
 5. ```Right x``` - where x is the number of metres the drone should move right
 6. ```Turn Left``` - Turns drone left by 90 degrees
 7. ```Turn Right``` - Turns drone right by 90 degrees
 8. ```Land``` - Lands and disarms the drone, and disconnects drone

<h3>controller.py</h3>
Defines commands for the drone to run.

<h3>connection.py</h3>
Connection logic for connection brdige between Rasberry Pi 5 and Pix Hawk 4, via telemetry radio.

<h3>parser.py</h3>
Supposed to parse output from stt into commands.

<h3>VoiceControlML</h3>

(Note: this version of the program has not been tested on the Pi 5 yet.)

```streamaudio23032026.py``` listens to the mic and uses a trained model to classify incoming speech as commands.
To test, 
  1) Download ```soundclassifier_with_metadata.tflite``` and ```streamaudio23032026.py```, and store them in the same folder.
  2) Run ```streamaudio23032026.py```.

This model is trained on my (Hao En's) voice, and in a quiet area. This may result in poor accuracy when used by someone else. Additional training from the pilot is required.

To actually make it do something, there's a bunch of if statements below ```line 137``` to put in your commands. Do note that, as of now, they will trigger several times a second.
