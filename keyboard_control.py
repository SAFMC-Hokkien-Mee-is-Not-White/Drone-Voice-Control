import asyncio
import logging
from Drone.controller import connect_vehicle
from NLP.commands import execute

logging.getLogger("mavsdk").setLevel(logging.CRITICAL)

async def keyboard_loop():
    """
    Simple command loop to control the drone via keyboard input.
    Type commands like: ARM, FORWARD 2, UP 1, YAW_LEFT 90 etc.
    Type QUIT to exit.
    """
    drone = await connect_vehicle("serial:///dev/ttyUSB0:57600")
    print("Enter commands (type QUIT to exit)")

    try:
        while True:
            # asyncio-safe input (runs blocking input() in a thread)
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, ">>> "
            )
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.upper() == "QUIT":
                print("[INFO] Exiting keyboard control")
                break

            parts = user_input.split()
            cmd = parts[0].upper()
            distance = None

            if len(parts) > 1:
                try:
                    distance = float(parts[1])
                except ValueError:
                    print("[WARN] Invalid distance, ignoring")

            print(f"[INPUT] Command: {cmd}, Distance: {distance}")
            await execute(drone, cmd, distance)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard loop interrupted")

    finally:
        # No explicit close() in MAVSDK — just signal disconnect
        print("[INFO] Vehicle disconnected")

if __name__ == "__main__":
    asyncio.run(keyboard_loop())
