import json
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
from google import genai
from google.genai import types

# 1. API Configuration
MY_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
client = genai.Client(api_key=MY_API_KEY)

IMAGE_FILENAME = "sharpener_far.jpg" 

# 2. Fix image rotation
raw_img = Image.open(IMAGE_FILENAME)
full_img = ImageOps.exif_transpose(raw_img)

full_img_byte_arr = io.BytesIO()
full_img.save(full_img_byte_arr, format='JPEG')
full_image_bytes = full_img_byte_arr.getvalue()

# =====================================================================
# STEP 1: Find the Sharpener (Bounding Box) in the Wide Shot
# =====================================================================
print("Step 1: Finding the sharpener in the full image...")
box_prompt = """
    Return a bounding box as a JSON array with labels. 
    Find only the "pencil sharpener".
    The format should be exactly: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "sharpener"}] 
    normalized to 0-1000.
"""

box_response = client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[
        types.Part.from_bytes(data=full_image_bytes, mime_type='image/jpeg'),
        box_prompt
    ],
    config=types.GenerateContentConfig(
        temperature=0.1, 
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)

# Parse the box coordinates
clean_box_json = box_response.text.strip().strip("`").replace("json", "").strip()
box_data = json.loads(clean_box_json)[0]
ymin, xmin, ymax, xmax = box_data["box_2d"]

# Convert normalized coordinates to actual image pixels
top = (ymin / 1000) * full_img.height
left = (xmin / 1000) * full_img.width
bottom = (ymax / 1000) * full_img.height
right = (xmax / 1000) * full_img.width

# Add a 10% padding to the bounding box so we don't cut off the edges of the sharpener
pad_y = (bottom - top) * 0.1
pad_x = (right - left) * 0.1
crop_box = (
    max(0, left - pad_x), 
    max(0, top - pad_y), 
    min(full_img.width, right + pad_x), 
    min(full_img.height, bottom + pad_y)
)

# =====================================================================
# STEP 2: Crop (Zoom) the image programmatically
# =====================================================================
print("Step 2: Cropping (zooming) into the sharpener...")
zoomed_img = full_img.crop(crop_box)

zoomed_byte_arr = io.BytesIO()
zoomed_img.save(zoomed_byte_arr, format='JPEG')
zoomed_image_bytes = zoomed_byte_arr.getvalue()

# =====================================================================
# STEP 3: Find specific parts on the newly ZOOMED image
# =====================================================================
print("Step 3: Finding specific parts on the zoomed image...")
parts_queries = ["hole for pencil", "metal blade", "screw"]
parts_prompt = f"""
    Locate the exact points for the following specific parts: {', '.join(parts_queries)}.
    The answer MUST follow this exact JSON format:
    [
      {{"point": [y, x], "label": "part name"}}
    ]
    The points are in [y, x] format normalized to 0-1000.
"""

parts_response = client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[
        types.Part.from_bytes(data=zoomed_image_bytes, mime_type='image/jpeg'),
        parts_prompt
    ],
    config=types.GenerateContentConfig(
        temperature=0.1, 
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)

print("\n--- FINAL PARTS JSON OUTPUT ---")
print(parts_response.text)
print("-------------------------------\n")

# =====================================================================
# STEP 4: Plot BOTH images side-by-side
# =====================================================================
clean_parts_json = parts_response.text.strip().strip("`").replace("json", "").strip()
parts_data = json.loads(clean_parts_json)

# Create a figure with 2 subplots (side-by-side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --- Plot 1: Full Image with Bounding Box ---
ax1.imshow(full_img)
ax1.set_title("1. Original Image (Sharpener Detected)")
rect = patches.Rectangle((left, top), right - left, bottom - top, 
                         linewidth=3, edgecolor='#facc15', facecolor='none')
ax1.add_patch(rect)
ax1.text(left, top - 10, "Target: Sharpener", color="black", backgroundcolor="#facc15", fontsize=12, weight='bold')
ax1.axis('off')

# --- Plot 2: Zoomed Image with Parts Dots ---
ax2.imshow(zoomed_img)
ax2.set_title("2. Zoomed Image (Parts Identified)")

for item in parts_data:
    y, x = item["point"]
    # Convert points based on the NEW zoomed image dimensions
    px, py = (x / 1000) * zoomed_img.width, (y / 1000) * zoomed_img.height
    
    ax2.plot(px, py, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax2.text(px + 10, py, item["label"], color="white", backgroundcolor="#ef4444", 
             fontsize=11, weight='bold', bbox=dict(facecolor='#ef4444', edgecolor='none', boxstyle='round,pad=0.3'))
             
ax2.axis('off')

plt.tight_layout()
plt.show()

