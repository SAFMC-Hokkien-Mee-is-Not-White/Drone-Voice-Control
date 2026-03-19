import asyncio
import math
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from config import TAKEOFF_ALTITUDE

# Global State
current_velocity = [0.0, 0.0, 0.0]  # vx, vy, vz (body frame)
current_yaw_rate = 0.0
offboard_running = False
_drone_ref = None  # set on connect

# Connection
async def connect_vehicle(connection_string):
    print("[INFO] Connecting...")
    drone = System()
    await drone.connect(system_address=connection_string)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[INFO] Connected to vehicle")
            break

    global _drone_ref
    _drone_ref = drone
    return drone

# Offboard Stream
async def offboard_loop(drone):
    global current_velocity, current_yaw_rate, offboard_running

    print("[INFO] OFFBOARD stream started")

    while offboard_running:
        try:
            vx, vy, vz = current_velocity
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vx, vy, vz, math.degrees(current_yaw_rate))
            )
        except Exception as e:
            print(f"[WARN] Offboard stream error: {e}")
            await asyncio.sleep(0.1)
            continue
        await asyncio.sleep(0.05)

async def start_offboard_stream(drone):
    global offboard_running

    offboard_running = True

    # Prime with a zero setpoint before enabling offboard
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )

    # Start stream task
    asyncio.create_task(offboard_loop(drone))

    print("[INFO] Priming OFFBOARD stream...")
    await asyncio.sleep(2)

def stop_offboard_stream():
    global offboard_running
    offboard_running = False

# Control Primitives
def set_velocity(vx, vy, vz):
    global current_velocity
    current_velocity = [vx, vy, vz]

def set_yaw_rate(rate):
    global current_yaw_rate
    current_yaw_rate = rate

# Arm & Takeoff
async def arm_and_takeoff(drone, altitude=TAKEOFF_ALTITUDE):
    print("[INFO] Starting takeoff sequence")

    await start_offboard_stream(drone)

    # Arm with retries
    print("[INFO] Arming...")
    armed = False
    for attempt in range(5):
        try:
            await drone.action.arm()
            print("[INFO] Armed!")
            armed = True
            break
        except Exception as e:
            print(f"[WARN] Arm attempt {attempt + 1}/5 failed: {e}")
            await asyncio.sleep(1.0)

    if not armed:
        print("[ERROR] All arm attempts failed")
        stop_offboard_stream()
        return False

    print(f"[INFO] Sending NAV_TAKEOFF...")
    try:
        await drone.action.takeoff()
        print("[INFO] Takeoff command sent")
    except Exception as e:
        print(f"[ERROR] Takeoff failed: {e}")
        stop_offboard_stream()
        return False

    async for position in drone.telemetry.position():
        alt = position.relative_altitude_m
        print(f"[INFO] Altitude: {alt:.1f}m", end="\r")
        if alt >= altitude * 0.01:
            break

    print(f"\n[INFO] Reached {altitude}m, switching to OFFBOARD...")

    try:
        await drone.offboard.start()
        print("[INFO] OFFBOARD mode active — ready for commands")
    except OffboardError as e:
        print(f"[ERROR] OFFBOARD rejected: {e}")
        stop_offboard_stream()
        return False

    return True

# Movement & Yaw
async def move_forward(drone, distance, speed=1.0):
    print(f"[INFO] Moving forward {distance}m")
    set_velocity(speed, 0, 0)
    await asyncio.sleep(distance / speed)
    set_velocity(0, 0, 0)
    await asyncio.sleep(0.5)

async def move_backward(drone, distance, speed=1.0):
    print(f"[INFO] Moving backward {distance}m")
    set_velocity(-speed, 0, 0)
    await asyncio.sleep(distance / speed)
    set_velocity(0, 0, 0)
    await asyncio.sleep(0.5)

async def move_left(drone, distance, speed=1.0):
    print(f"[INFO] Moving left {distance}m")
    set_velocity(0, -speed, 0)
    await asyncio.sleep(distance / speed)
    set_velocity(0, 0, 0)
    await asyncio.sleep(0.5)

async def move_right(drone, distance, speed=1.0):
    print(f"[INFO] Moving right {distance}m")
    set_velocity(0, speed, 0)
    await asyncio.sleep(distance / speed)
    set_velocity(0, 0, 0)
    await asyncio.sleep(0.5)

async def move_up(drone, distance, speed=1.0):
    print(f"[INFO] Moving up {distance}m")
    set_velocity(0, 0, -speed)
    await asyncio.sleep(distance / speed)
    set_velocity(0, 0, 0)
    await asyncio.sleep(0.5)

async def move_down(drone, distance, speed=1.0):
    print(f"[INFO] Moving down {distance}m")
    set_velocity(0, 0, speed)
    await asyncio.sleep(distance / speed)
    set_velocity(0, 0, 0)
    await asyncio.sleep(0.5)

async def yaw_left(drone, angle=90, rate=30):
    print(f"[INFO] Yawing left {angle} degrees")
    yaw_rate_rad = rate * (math.pi / 180)
    duration = angle / rate
    set_yaw_rate(-yaw_rate_rad)
    await asyncio.sleep(duration)
    set_yaw_rate(0)
    await asyncio.sleep(0.5)

async def yaw_right(drone, angle=90, rate=30):
    print(f"[INFO] Yawing right {angle} degrees")
    yaw_rate_rad = rate * (math.pi / 180)
    duration = angle / rate
    set_yaw_rate(yaw_rate_rad)
    await asyncio.sleep(duration)
    set_yaw_rate(0)
    await asyncio.sleep(0.5)

# Landing & Disarm
async def land(drone):
    print("[INFO] Landing...")

    stop_offboard_stream()
    await asyncio.sleep(0.1)

    try:
        await drone.offboard.stop()
    except Exception:
        pass

    await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("[INFO] Landed!")
            break

    await drone.action.disarm()
    print("[INFO] Disarmed")
