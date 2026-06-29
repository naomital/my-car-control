import cv2
import socket
import time

from depth_runtime import DepthEstimator
from config import CONFIG


CAR_IP = "172.16.11.1"
CONTROL_PORT = 23458
RTSP_URL = "rtsp://172.16.11.1/live/ch00_1"

BASE_PACKET = bytearray.fromhex(
    "ca 47 d5 00 00 00 00 00 66 80 80 80 00 00 80 99"
)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Steering/throttle byte values -- tune these in config.py (CONFIG.steering / CONFIG.throttle).
CENTER = CONFIG.steering.center
STEERING_STEP = CONFIG.steering.step
STEERING_MIN = CONFIG.steering.min
STEERING_MAX = CONFIG.steering.max

THROTTLE_STOP = CONFIG.throttle.stop
THROTTLE_FORWARD = CONFIG.throttle.forward
THROTTLE_BACKWARD = CONFIG.throttle.backward


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def send_drive(steering, throttle):
    steering = clamp(steering, 0, 255)
    throttle = clamp(throttle, 0, 255)

    packet = BASE_PACKET[:]
    packet[9] = steering
    packet[10] = throttle
    packet[14] = packet[9] ^ packet[10] ^ packet[11]

    sock.sendto(packet, (CAR_IP, CONTROL_PORT))


def stop_and_center():
    for _ in range(10):
        send_drive(CENTER, THROTTLE_STOP)
        time.sleep(0.05)


def smooth_center_steering(current_steering, throttle, duration_sec=1.0, steps=20):
    """Return steering to center gradually over ~1 second."""
    current_steering = clamp(current_steering, STEERING_MIN, STEERING_MAX)
    if steps <= 0:
        send_drive(CENTER, throttle)
        return CENTER

    for i in range(1, steps + 1):
        blend = i / steps
        next_steering = int(round(current_steering + (CENTER - current_steering) * blend))
        send_drive(next_steering, throttle)
        time.sleep(duration_sec / steps)
    return CENTER


def main():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Could not open video stream")
        print("Try VLC:", RTSP_URL)
        return

    print("Loading Depth Anything V2 (vits) ...")
    depth_estimator = DepthEstimator()  # smallest checkpoint, threaded, latest-frame-only
    depth_estimator.start()
    print(f"Depth model ready on device={depth_estimator.device}")

    steering = CENTER
    throttle = THROTTLE_STOP

    print("Controls:")
    print("A = more left")
    print("D = more right")
    print("C = center steering")
    print("Z = smooth center steering (1 sec)")
    print("W = forward with current steering")
    print("S = backward with current steering")
    print("SPACE = stop throttle only")
    print("X = stop + center")
    print("Q / ESC = quit")
    print()
    print("IMPORTANT: click the video window and make sure keyboard is English.")

    last_send = 0
    send_interval = 0.05  # 20Hz

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("No frame")
                break

            frame = cv2.resize(frame, (640, 360))

            # Hand the latest camera frame to the depth model. Non-blocking:
            # the background thread just swaps in this frame as its next
            # target, dropping whatever it hadn't gotten to yet -- the
            # control loop never waits on inference.
            depth_estimator.submit_frame(frame)
            depth_result = depth_estimator.get_latest()

            cv2.putText(
                frame,
                f"steering={steering} throttle={throttle}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "A/D steer | C center | Z smooth-center | W/S drive | SPACE stop | X full stop",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            if depth_result is not None:
                # visualization is grayscale (brighter = closer) with the
                # region driving an imminent collision warning painted red.
                depth_vis = depth_result.visualization.copy()
                cv2.putText(
                    depth_vis,
                    f"depth {depth_result.fps:.1f} FPS ({depth_result.inference_seconds * 1000:.0f} ms)",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if depth_result.collision_warning:
                    cv2.putText(
                        frame, "!! OBSTACLE AHEAD !!", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    )
                display = cv2.hconcat([frame, depth_vis])
            else:
                display = frame

            cv2.imshow("WL Car Live Control - CLICK HERE", display)

            key = cv2.waitKey(1)
            print(key)

            if key != -1:
                key = key & 0xFF
                print("Key pressed:", key, chr(key) if 32 <= key <= 126 else "")

                if key == 27 or key == ord("q"):
                    break

                elif key == ord("a"):
                    steering = clamp(steering - STEERING_STEP, STEERING_MIN, STEERING_MAX)
                    print("left, steering =", steering)
                    send_drive(steering, throttle)

                elif key == ord("d"):
                    steering = clamp(steering + STEERING_STEP, STEERING_MIN, STEERING_MAX)
                    print("right, steering =", steering)
                    send_drive(steering, throttle)

                elif key == ord("c"):
                    steering = CENTER
                    print("center")
                    send_drive(steering, throttle)

                elif key == ord("z"):
                    print("smooth center (1 sec)")
                    steering = smooth_center_steering(steering, throttle, duration_sec=1.0, steps=20)

                elif key == ord("w"):
                    throttle = THROTTLE_FORWARD
                    print("forward, throttle =", throttle, "steering =", steering)
                    send_drive(steering, throttle)

                elif key == ord("s"):
                    throttle = THROTTLE_BACKWARD
                    print("backward, throttle =", throttle, "steering =", steering)
                    send_drive(steering, throttle)

                elif key == ord(" "):
                    throttle = THROTTLE_STOP
                    print("stop throttle, keep steering =", steering)
                    send_drive(steering, throttle)

                elif key == ord("x"):
                    steering = CENTER
                    throttle = THROTTLE_STOP
                    print("full stop + center")
                    send_drive(steering, throttle)

            # heartbeat: keep sending last state
            now = time.time()
            if now - last_send > send_interval:
                send_drive(steering, throttle)
                last_send = now

    finally:
        print("Stopping...")
        stop_and_center()
        depth_estimator.stop()
        cap.release()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()