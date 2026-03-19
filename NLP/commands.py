from Drone.controller import *
from config import SPEED, YAW_RATE

async def execute(drone, command, value):
    print(f"[EXEC] {command} | value={value}")

    if command == "ARM":
        await arm_and_takeoff(drone, altitude=3)
    elif command == "FORWARD":
        dist = value if value else 2
        await move_forward(drone, dist, SPEED)
    elif command == "BACKWARD":
        dist = value if value else 2
        await move_backward(drone, dist, SPEED)
    elif command == "LEFT":
        dist = value if value else 2
        await move_left(drone, dist, SPEED)
    elif command == "RIGHT":
        dist = value if value else 2
        await move_right(drone, dist, SPEED)
    elif command == "UP":
        dist = value if value else 2
        await move_up(drone, dist, SPEED)
    elif command == "DOWN":
        dist = value if value else 1
        await move_down(drone, dist, SPEED)
    elif command == "YAW_LEFT":
        angle = value if value else 90
        await yaw_left(drone, angle, YAW_RATE)
    elif command == "YAW_RIGHT":
        angle = value if value else 90
        await yaw_right(drone, angle, YAW_RATE)
    elif command == "STOP":
        set_velocity(0, 0, 0)
        set_yaw_rate(0)
    elif command == "LAND":
        await land(drone)
    else:
        print("[WARN] Unknown command")
