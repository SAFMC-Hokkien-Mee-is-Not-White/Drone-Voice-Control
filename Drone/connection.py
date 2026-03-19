import asyncio
import logging
from mavsdk import System

logging.getLogger("mavsdk").setLevel(logging.CRITICAL)

async def connect_vehicle(connection_string):
    print("[INFO] Connecting...")
    drone = System(sysid=255, compid=240)  # GCS identity — trusted by PX4
    await drone.connect(system_address=connection_string)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[INFO] Connected to vehicle")
            break

    # Wait for PX4 to be fully ready before sending any commands
    print("[INFO] Waiting for vehicle ready...")
    await asyncio.sleep(3.0)

    global _drone_ref
    _drone_ref = drone
    return drone

async def disconnect_vehicle(drone):
    # MAVSDK has no explicit close(); just let the coroutine/event loop end.
    # If you need a clean signal, cancel any running tasks.
    print("[INFO] Disconnecting vehicle...")
    print("[INFO] Vehicle disconnected")

async def main():
    drone = await connect_vehicle()
    # ... do work ...
    await disconnect_vehicle(drone)

if __name__ == "__main__":
    asyncio.run(main())
