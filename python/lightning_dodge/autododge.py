#!/usr/bin/python3
import sys
import signal
import time
import Xlib.display
import PIL.Image
import PIL.ImageStat
from pynput import keyboard

def sample_screen(x, y, w, h):
    # Capture screen
    root = Xlib.display.Display().screen().root
    image = root.get_image(x, y, w, h, Xlib.X.ZPixmap, 0xffffffff)
    image_raw = image.data

    # Sometimes, image data will be a string, I don't get why, but we always want bytes
    if isinstance(image_raw, str):
        image_raw = str.encode(image_raw)

    # Convert raw data to a PIL image in HSV format
    return PIL.Image.frombytes("RGB", (w, h), image_raw, "raw", "BGRX").convert("HSV")


def probe_lightness():
    # Sample screen along probe
    screen = Xlib.display.Display().screen()
    width = screen.width_in_pixels
    height = screen.height_in_pixels
    image_hsv = sample_screen(0, height//2, width//2, 1)

    # Get normalized median HSV color
    median_color = PIL.ImageStat.Stat(image_hsv).median
    median_color = [x/255 for x in median_color]

    # Calculate lightness
    return median_color[2]*(1-median_color[1]/2)


def hit(kb, key, interval):
    kb.press(key)
    time.sleep(interval)
    kb.release(key)


def handler(signum, frame):
    print("Exiting now")
    exit(1)


active = False
def main(argv):

    # Toggle bot by pressing the '!' key
    def on_press(key):
        if hasattr(key, 'char'):
            if key.char == '!':
                global active
                active = not active
                print(f'Active: {active}')

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Declare our keyboard interface
    kb = keyboard.Controller()
    signal.signal(signal.SIGINT, handler)
    
    state = False
    old_state = False
    count = 0
    while True:
        if active:
            L = probe_lightness()
            state = L > 0.75
            if state and not old_state:
                count += 1
                hit(kb, 'e', 0.15)
                print(f'Dodging lightning strike #{count}')
                
            old_state = state
        time.sleep(0.05)


if __name__ == '__main__':
    main(sys.argv[1:])