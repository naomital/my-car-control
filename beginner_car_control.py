"""
==============================================================================
 BEGINNER'S GUIDE: Connecting to and Driving a WiFi RC Car
==============================================================================

WHO THIS IS FOR
----------------
You just got a small WiFi-controlled RC car (the kind that streams video over
RTSP and accepts drive commands over UDP) and you want to control it from
your computer with Python. This file is a *standalone*, heavily-commented
starting point. It does NOT do any computer vision, depth estimation, or
autonomous navigation -- it only covers the three things you need before any
of that is possible:

    1. Connecting to the car over WiFi (sockets, IP addresses, ports)
    2. Reading the car's live camera feed (RTSP video stream)
    3. Sending steering / throttle commands to the car (UDP packets)

Once you're comfortable with everything in this file, you're ready to start
layering on computer vision (e.g. object detection, depth estimation) and
real autonomy. This repo's more advanced scripts (e.g. connect_wifi_car.py)
build on exactly this foundation by adding a depth-estimation model in the
loop -- but that is a separate, later step. Don't worry about it yet.

HOW THESE CARS WORK (the big picture)
--------------------------------------
Most of these budget WiFi RC cars work like this:

  - The car itself runs a small WiFi access point (or joins your WiFi
    network). Once connected, your computer and the car are on the same
    local network and can talk to each other using IP addresses, exactly
    like two computers talking over the internet -- just on a tiny private
    network instead of the public internet.

  - The car has a fixed IP address on that network (commonly something like
    192.168.x.1 or 172.16.x.1 -- check your car's manual/app, or use a
    network scanner like `arp -a` on your computer after connecting).

  - The car exposes a VIDEO stream using RTSP (Real Time Streaming Protocol)
    -- think of it as a "live video URL" your computer can open, the same
    way VLC media player can open a network video stream. OpenCV
    (cv2.VideoCapture) can read RTSP streams directly, just like a video
    file.

  - The car listens for CONTROL commands on a UDP port. UDP is a simple,
    fast, "fire and forget" network protocol -- you send a small packet of
    bytes, the car reads it, and the car does not have to confirm receipt
    back to you (unlike TCP, which guarantees delivery). This is perfect
    for real-time control: if a packet is lost, you simply send another one
    a few milliseconds later, so it doesn't matter.

  - The exact bytes inside that UDP packet form a tiny "protocol" specific
    to this car model: a fixed-size sequence of bytes where certain byte
    positions (offsets) mean "steering value" or "throttle value", plus a
    checksum byte so the car can detect a corrupted packet. This protocol
    was reverse-engineered for THIS car; if you have a different model, the
    packet layout will likely be different (see "Adapting this to your own
    car" near the bottom).

WHAT YOU NEED BEFORE RUNNING THIS
-----------------------------------
  1. Python 3.8+ installed.
  2. OpenCV for Python:      pip install opencv-python
  3. The car powered on, broadcasting/joined to WiFi, and your computer
     connected to that SAME WiFi network (check this first if anything
     fails below -- it's the most common issue).
  4. The car's IP address and control port. If you don't already know them:
       - Check the car's manual / companion phone app.
       - Or, with the car connected, run a tool like Wireshark or
         `arp -a` (Windows) / `arp -a` (macOS/Linux) to see devices on
         your local network and guess from there.
       - Or use a network port scanner (e.g. `nmap`) to find an open UDP
         port on the car's IP, then experiment carefully.
  5. The car's RTSP video URL. Often follows a pattern like:
       rtsp://<car_ip>/live/ch00_1
     but check your car's documentation -- this varies by manufacturer.

QUICK START
-----------
    python beginner_car_control.py

Then, with the video window focused (click on it first!) and your keyboard
in English layout:

    A / D      steer left / right
    C          center the steering
    W          drive forward (uses current steering)
    S          drive backward (uses current steering)
    SPACE      stop the throttle (steering unchanged)
    X          full stop + re-center steering
    P          save a screenshot of the current camera frame to ./assets
    Q / ESC    quit (also stops the car safely before exiting)

==============================================================================
"""

from __future__ import annotations

import os
import socket
import time

import cv2

# Where "P" (save screenshot) writes images -- handy for grabbing a real
# example of the car's camera feed for documentation (e.g. a project README).
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# =============================================================================
# STEP 1: Network settings -- WHO are we talking to, and HOW?
# =============================================================================
#
# These three values are the absolute minimum you need to know about your
# car before writing a single line of control code:
#
#   CAR_IP        -- the car's IP address on your local WiFi network.
#   CONTROL_PORT  -- the UDP port the car listens on for drive commands.
#   RTSP_URL      -- the network video stream URL for the car's camera.
#
# If you're not sure these are correct for your hardware, that's the very
# first thing to verify -- nothing else in this file will work otherwise.

CAR_IP = "172.16.11.1"
CONTROL_PORT = 23458
RTSP_URL = "rtsp://172.16.11.1/live/ch00_1"


# =============================================================================
# STEP 2: The control packet -- WHAT do we actually send?
# =============================================================================
#
# This car expects a fixed 16-byte UDP packet for every drive command. Most
# of the bytes are constant "header" bytes specific to this car's protocol
# (don't worry about what each one means -- they were reverse-engineered by
# observing the official app's network traffic). The only bytes that change
# from packet to packet are:
#
#   packet[9]   -- steering value (0-255). 0x80 (=128) is "centered".
#                  Smaller values steer one way, larger values the other.
#   packet[10]  -- throttle value (0-255). 0x80 (=128) is "stopped".
#                  Values above that drive forward, values below drive
#                  backward (or vice versa -- verify empirically with YOUR
#                  car at low speed in a safe, open space).
#   packet[14]  -- a checksum byte. This car computes it as the XOR of
#                  bytes 9, 10 and 11. If the checksum doesn't match what
#                  the car expects, it will likely ignore the packet, so
#                  recompute it every time you change steering/throttle.
#
# "XOR checksum" just means: take the bytes you want to protect and combine
# them with the bitwise XOR (^) operator. It's a cheap way for the receiver
# to detect simple transmission errors -- not real security, just sanity
# checking.

BASE_PACKET = bytearray.fromhex(
    "ca 47 d5 00 00 00 00 00 66 80 80 80 00 00 80 99"
)

# Byte offsets inside BASE_PACKET, named so the rest of the code never uses
# "magic numbers" -- if you reverse-engineer a different car's protocol and
# the offsets differ, this is the only place you need to change.
STEERING_BYTE_OFFSET = 9
THROTTLE_BYTE_OFFSET = 10
CHECKSUM_BYTE_OFFSET = 14
CHECKSUM_INPUT_OFFSETS = (9, 10, 11)  # bytes XORed together to make the checksum


# =============================================================================
# STEP 3: Steering / throttle values -- tune these for YOUR car
# =============================================================================
#
# These are just *suggested* starting points. Every individual car (even the
# same model) can behave slightly differently, so test these gently, with
# the car raised off the ground or in a wide-open space, before trusting
# them at full speed.

CENTER = 0x80          # straight steering
STEERING_STEP = 4      # how much each key-press changes steering by
STEERING_MIN = 0x40    # full lock to one side
STEERING_MAX = 0xC0    # full lock to the other side

THROTTLE_STOP = 0x80       # motor off
THROTTLE_FORWARD = 0x8A    # gentle forward -- raise this slowly to go faster
THROTTLE_BACKWARD = 0x76   # gentle backward


# =============================================================================
# STEP 4: Helper functions
# =============================================================================

def clamp(value: int, low: int, high: int) -> int:
    """Keep `value` inside [low, high]. Used so steering/throttle never
    accidentally leave the range the car expects (0-255), which could
    otherwise wrap around to a nonsensical byte value."""
    return max(low, min(high, value))


def build_packet(steering: int, throttle: int) -> bytearray:
    """Take the constant BASE_PACKET template and stamp in the current
    steering/throttle values plus a fresh checksum. Returns a brand-new
    bytearray each time so we never accidentally mutate BASE_PACKET itself."""
    steering = clamp(steering, 0, 255)
    throttle = clamp(throttle, 0, 255)

    packet = BASE_PACKET[:]  # shallow copy -- BASE_PACKET stays untouched
    packet[STEERING_BYTE_OFFSET] = steering
    packet[THROTTLE_BYTE_OFFSET] = throttle

    checksum = 0
    for offset in CHECKSUM_INPUT_OFFSETS:
        checksum ^= packet[offset]
    packet[CHECKSUM_BYTE_OFFSET] = checksum

    return packet


def send_drive(sock: socket.socket, steering: int, throttle: int) -> None:
    """Build a drive packet and fire it at the car over UDP.

    Note this is "fire and forget": sendto() returns as soon as the packet
    has been handed to the OS's network stack. We get no confirmation the
    car actually received or acted on it. That's normal for UDP and fine
    for real-time control -- we simply keep sending updated packets, so a
    single dropped packet has no lasting effect (see the "heartbeat" loop
    below)."""
    packet = build_packet(steering, throttle)
    sock.sendto(packet, (CAR_IP, CONTROL_PORT))


def stop_and_center(sock: socket.socket) -> None:
    """Safety helper: send "stop + centered steering" several times in a
    row. We send it repeatedly (not just once) because UDP packets can be
    silently dropped -- repeating a few times over ~0.5s makes it very
    likely at least one arrives, which matters most exactly when you're
    trying to stop the car."""
    for _ in range(10):
        send_drive(sock, CENTER, THROTTLE_STOP)
        time.sleep(0.05)


# =============================================================================
# STEP 5: Main loop -- camera + keyboard + drive commands, all together
# =============================================================================

def main() -> None:
    # -- 5a. Open the car's camera stream ------------------------------------
    # cv2.VideoCapture can open an RTSP URL exactly like a local video file.
    # CAP_PROP_BUFFERSIZE=1 tells OpenCV/FFmpeg to keep at most 1 frame
    # buffered internally -- without this, on a slow/laggy network OpenCV
    # can silently queue up old frames, and you end up watching video that's
    # several seconds behind reality. For a *live* control loop we always
    # want the newest frame, even if that means occasionally dropping one.
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Could not open the video stream.")
        print(f"  RTSP URL tried: {RTSP_URL}")
        print("  Troubleshooting:")
        print("   1. Is your computer connected to the car's WiFi network?")
        print("   2. Is CAR_IP correct for your car?")
        print("   3. Try opening the same RTSP_URL directly in VLC Media")
        print("      Player (Media > Open Network Stream) -- if VLC also")
        print("      can't connect, the problem is networking, not Python.")
        return

    # A UDP socket doesn't need to "connect" first the way TCP does -- we
    # just create it once and use sendto() with a destination each time.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Current commanded state. We keep these as variables (not just sending
    # once per key press) because of the "heartbeat" pattern explained
    # below.
    steering = CENTER
    throttle = THROTTLE_STOP

    print("Connected. Controls:")
    print("  A / D      steer left / right")
    print("  C          center steering")
    print("  W          drive forward")
    print("  S          drive backward")
    print("  SPACE      stop throttle (keep steering)")
    print("  X          full stop + center")
    print("  P          save a screenshot of the camera feed to ./assets")
    print("  Q / ESC    quit")
    print()
    print("IMPORTANT: click the video window first so it has keyboard focus,")
    print("and make sure your OS keyboard layout is set to English.")

    # -- 5b. The "heartbeat" pattern ------------------------------------------
    # Many of these cars expect to hear from you regularly, and will stop
    # the motors on their own as a safety feature if they don't receive any
    # packet for a short time (so the car doesn't drive off into a wall if
    # your WiFi or your program crashes). So instead of sending a command
    # only when a key is pressed, we resend the *current* steering/throttle
    # at a fixed rate (here, 20 times per second) for as long as the
    # program runs. This is "send_interval" below.
    last_send_time = 0.0
    send_interval_seconds = 0.05  # 1 / 0.05 = 20 packets per second

    try:
        while True:
            # -- read one frame from the camera --
            ok, frame = cap.read()
            if not ok:
                print("Lost the video frame (network hiccup or stream ended).")
                break

            # Resize for a consistent, predictable display size regardless
            # of the car's native camera resolution.
            frame = cv2.resize(frame, (640, 360))

            # -- on-screen status text, just for your own visibility --
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
                "A/D steer | C center | W/S drive | SPACE stop | X full stop",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Beginner WiFi Car Control - CLICK HERE", frame)

            # -- read a keyboard key, if any (waits at most 1ms) --
            key = cv2.waitKey(1)
            if key != -1:
                key = key & 0xFF  # normalize to a plain ASCII byte

                if key == 27 or key == ord("q"):  # ESC or Q
                    break

                elif key == ord("a"):
                    steering = clamp(steering - STEERING_STEP, STEERING_MIN, STEERING_MAX)
                    send_drive(sock, steering, throttle)

                elif key == ord("d"):
                    steering = clamp(steering + STEERING_STEP, STEERING_MIN, STEERING_MAX)
                    send_drive(sock, steering, throttle)

                elif key == ord("c"):
                    steering = CENTER
                    send_drive(sock, steering, throttle)

                elif key == ord("w"):
                    throttle = THROTTLE_FORWARD
                    send_drive(sock, steering, throttle)

                elif key == ord("s"):
                    throttle = THROTTLE_BACKWARD
                    send_drive(sock, steering, throttle)

                elif key == ord(" "):
                    throttle = THROTTLE_STOP
                    send_drive(sock, steering, throttle)

                elif key == ord("x"):
                    steering = CENTER
                    throttle = THROTTLE_STOP
                    send_drive(sock, steering, throttle)

                elif key == ord("p"):
                    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
                    filename = f"camera_feed_{int(time.time())}.png"
                    path = os.path.join(SCREENSHOT_DIR, filename)
                    cv2.imwrite(path, frame)
                    print(f"Saved screenshot: {path}")

            # -- heartbeat: resend current state even if no key was pressed --
            now = time.time()
            if now - last_send_time > send_interval_seconds:
                send_drive(sock, steering, throttle)
                last_send_time = now

    finally:
        # No matter how we exit (normal quit, error, Ctrl+C), always try to
        # leave the car stopped and centered rather than driving away
        # unattended.
        print("Stopping the car and cleaning up...")
        stop_and_center(sock)
        cap.release()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()


# =============================================================================
# ADAPTING THIS TO YOUR OWN CAR
# =============================================================================
#
# If you have a *different* WiFi RC car model, the concepts in this file
# (UDP control packets, RTSP video, a heartbeat loop) very likely still
# apply -- but the specific numbers almost certainly won't. To adapt this
# file:
#
#   1. Find the car's IP address and control port. Connect your computer
#      to the car's WiFi, then use a packet sniffer (e.g. Wireshark) while
#      using the car's official phone app to drive it manually. Watch for
#      UDP packets being sent from your phone to the car -- that traffic
#      tells you the IP, port, and the exact byte layout to replicate here.
#
#   2. Find the RTSP video URL the same way (or check the app's settings
#      screen, which sometimes shows it directly).
#
#   3. Replace CAR_IP, CONTROL_PORT, RTSP_URL, BASE_PACKET, and the byte
#      offsets in STEP 2 to match what you observed.
#
#   4. Test steering/throttle changes ONE STEP AT A TIME with the car
#      raised off the ground, confirming each byte you change does what
#      you expect, before trusting any of it at full speed on the floor.
#
# WHERE TO GO FROM HERE
# =============================================================================
#
# Once driving the car manually like this feels comfortable, the natural
# next steps toward autonomy (covered by other, more advanced files in this
# repository) are:
#
#   1. Run a monocular depth-estimation model (e.g. Depth Anything V2) on
#      each camera frame to estimate "what's close" vs "what's far", even
#      with just a single regular camera (no special depth/stereo hardware
#      needed).
#   2. Turn that depth information into a simple obstacle/collision signal
#      (e.g. "something is closing in fast in front of the car").
#   3. Use that signal to automatically adjust steering/throttle instead of
#      (or in addition to) the keyboard, which is the first real step into
#      semi-autonomous driving.
#   4. Eventually, build a map of where the car has already been, so it can
#      find its way back to its starting point ("home") on its own.
#
# Each of those steps is a separate, self-contained problem -- there's no
# need to tackle them all at once. Get comfortable with THIS file first.
# =============================================================================
