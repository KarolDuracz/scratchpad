import mss
import imageio
from PIL import Image
import numpy as np
import time

# Define the size of the capture area
fps = 30
duration = 10  # Duration of the recording in seconds

# Calculate the number of frames to capture
num_frames = fps * duration

# Create a list to store the captured images
images = []

print("Starting capture...")

with mss.mss() as sct:
    # Get the dimensions of the virtual screen (combined screen area)
    monitors = sct.monitors
    monitor = monitors[1]  # `monitors[0]` is the whole screen, `monitors[1]` is the combined area

    for _ in range(num_frames):
        # Capture the entire extended screen area
        screenshot = sct.grab(monitor)
        
        # Convert to PIL Image
        screenshot = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        
        # Convert to NumPy array
        img_array = np.array(screenshot)
        
        # Append to the images list
        images.append(img_array)
        
        # Wait for the next frame
        time.sleep(1 / fps)

print("Capture complete. Saving GIF...")

# Save the images as a GIF
with imageio.get_writer('output.gif', mode='I', duration=1 / fps) as writer:
    for img in images:
        writer.append_data(img)

print("GIF saved as output.gif")

# version to record area 800x600
"""
import mss
import imageio
from PIL import Image
import numpy as np
import time

# Define the size of the capture area
width, height = 800, 600
fps = 30
duration = 10  # Duration of the recording in seconds

# Calculate the number of frames to capture
num_frames = fps * duration

# Define the capture area (you might need to adjust the position)
left = 0
top = 0

# Create a list to store the captured images
images = []

print("Starting capture...")

with mss.mss() as sct:
    for _ in range(num_frames):
        # Capture the screen area
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        
        # Convert to PIL Image
        screenshot = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        
        # Convert to NumPy array
        img_array = np.array(screenshot)
        
        # Append to the images list
        images.append(img_array)
        
        # Wait for the next frame
        time.sleep(1 / fps)

print("Capture complete. Saving GIF...")

# Save the images as a GIF
with imageio.get_writer('output.gif', mode='I', duration=1 / fps) as writer:
    for img in images:
        writer.append_data(img)

print("GIF saved as output.gif")

"""
