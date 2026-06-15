Python 3.14.4 (tags/v3.14.4:23116f9, Apr  7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import json
... import io
... import matplotlib.pyplot as plt
... import matplotlib.patches as patches # <-- Added this to draw rectangles!
... from PIL import Image, ImageOps
... from google import genai
... from google.genai import types
... 
... # 1. API Configuration
... MY_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
... client = genai.Client(api_key=MY_API_KEY)
... 
... IMAGE_FILENAME = "sharpener_far.jpg"
... 
... # 2. Fix image rotation 
... raw_img = Image.open(IMAGE_FILENAME)
... img = ImageOps.exif_transpose(raw_img)
... 
... img_byte_arr = io.BytesIO()
... img.save(img_byte_arr, format='JPEG')
... image_bytes = img_byte_arr.getvalue()
... 
... # 3. The Bounding Box Prompt (Exactly as you found it)
... prompt = """
...     Return bounding boxes as a JSON array with labels. Never return masks
...     or code fencing. Limit to 25 objects. Include as many objects as you
...     can identify on the table.
...     If an object is present multiple times, name them according to their
...     unique characteristic (colors, size, position, unique characteristics, etc..).
...     The format should be as follows: [{"box_2d": [ymin, xmin, ymax, xmax],
...     "label": "label for the object"}] normalized to 0-1000. The values in
...     box_2d must only be integers
... """
... 
... # 4. Call the model
... print("Scanning image for objects and generating bounding boxes...")
... response = client.models.generate_content(
...     model="gemini-robotics-er-1.6-preview",
...     contents=[
...         types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
...         prompt
...     ],
...     config=types.GenerateContentConfig(
...         temperature=1.0, 
...         thinking_config=types.ThinkingConfig(thinking_budget=0)
...     )
... )
... 
... # 5. OUTPUT THE RAW JSON TEXT
... print("\n--- RAW JSON OUTPUT ---")
print(response.text)
print("-----------------------\n")

# 6. OUTPUT THE IMAGE WITH PLOTTED BOXES
fig, ax = plt.subplots(1, figsize=(12, 10))
ax.imshow(img)

try:
    # Clean up any markdown blocks
    clean_json = response.text.strip().strip("`").replace("json", "").strip()
    data = json.loads(clean_json)

    for item in data:
        # Extract the 4 box coordinates
        ymin, xmin, ymax, xmax = item["box_2d"]
        label = item["label"]
        
        # Convert normalized 0-1000 coordinates to actual image pixels
        ymin_px = (ymin / 1000) * img.height
        xmin_px = (xmin / 1000) * img.width
        ymax_px = (ymax / 1000) * img.height
        xmax_px = (xmax / 1000) * img.width
        
        # Calculate width and height of the box
        box_width = xmax_px - xmin_px
        box_height = ymax_px - ymin_px
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (xmin_px, ymin_px), box_width, box_height, 
            linewidth=3, edgecolor='#ef4444', facecolor='none'
        )
        
        # Add the rectangle to the image
        ax.add_patch(rect)
        
        # Add the label right above the box
        plt.text(xmin_px, ymin_px - 8, label, color="white", backgroundcolor="#ef4444", 
                 fontsize=12, weight='bold', bbox=dict(facecolor='#ef4444', edgecolor='none', boxstyle='round,pad=0.3'))

except Exception as e:
    print(f"Could not parse or draw the image. Error: {e}")

plt.axis('off')
plt.show()




