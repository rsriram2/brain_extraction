import os
import csv
import tkinter as tk
from PIL import Image, ImageTk

# Configuration
IMG_DIR = "/Users/rushil/ichseg/dockerctbet/new/image_ss_dockerctbet_failures"  # Set your directory
OUTPUT_FILE = "/Users/rushil/ichseg/dockerctbet/new/dockerctbetp_final.csv"
INITIAL_WIDTH, INITIAL_HEIGHT = 800, 600  # Set an initial window size
STATUS_HEIGHT = 50  # Fixed height for classification labels

print("📂 Checking image directory:", IMG_DIR)

# Load existing annotations
annotations = {}
try:
    if os.path.exists(OUTPUT_FILE):
        print("📄 Loading annotations from:", OUTPUT_FILE)
        with open(OUTPUT_FILE, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read header row
            for row in reader:
                annotations[row[0]] = row[1:]
        print(f"✅ Loaded {len(annotations)} annotated images.")
    else:
        print("⚠️ No existing annotations file found. Starting fresh.")
except Exception as e:
    print("❌ Error loading annotations:", str(e))

# Get image files, sort them, and find the first unclassified image
try:
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".png", ".jpg", ".jpeg"))])
    print(f"🖼 Found {len(img_files)} images in directory.")
except Exception as e:
    print("❌ Error reading image directory:", str(e))
    img_files = []

if not img_files:
    print("🚨 No images found. Exiting script.")
    exit(1)

first_unclassified_index = next((i for i, f in enumerate(img_files) if f not in annotations), 0)
img_index = first_unclassified_index  # Start with the first unclassified image
print(f"📌 Starting at image index {img_index}: {img_files[img_index]}")

# Classification state
classification_keys = [
    "1 - Volumetrics", "2 - Registration", "3 - DL"
]
classification = {key: "no" for key in classification_keys}

# UI setup
root = tk.Tk()
root.title("Image Review App")
root.geometry(f"{INITIAL_WIDTH}x{INITIAL_HEIGHT}")  # Set initial window size

print("✅ Tkinter UI initialized.")

# Main Frame to Hold Image & Status
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Image Frame (Takes most of the space)
img_frame = tk.Frame(main_frame, bg="black")
img_frame.pack(fill=tk.BOTH, expand=True)

img_label = tk.Label(img_frame, bg="black")
img_label.pack(fill=tk.BOTH, expand=True)

# Status Frame (Fixed at Bottom)
status_frame = tk.Frame(root, bg="white", padx=10, pady=5, height=STATUS_HEIGHT)
status_frame.pack(fill=tk.X, side=tk.BOTTOM)
status_frame.pack_propagate(False)  # Prevent shrinking

status_label = tk.Label(status_frame, text="", font=("Arial", 12), fg="black", bg="white", wraplength=700)
status_label.pack()

# Global variables
current_img = None
img_tk = None  # Ensure global reference to prevent garbage collection

# Load image and existing classification if available
def load_image():
    global img_index, current_img, img_tk

    print(f"🔄 Attempting to load image {img_index}/{len(img_files)}: {img_files[img_index]}")

    try:
        if img_index < 0:
            img_index = 0
        if img_index >= len(img_files):
            img_index = len(img_files) - 1

        img_path = os.path.join(IMG_DIR, img_files[img_index])
        print(f"🖼 Loading image from path: {img_path}")

        if not os.path.exists(img_path):
            print(f"❌ ERROR: Image file not found: {img_path}")
            return

        img = Image.open(img_path).convert("RGB")  # Convert to RGB to avoid mode issues
        current_img = img  # Store the new image

        update_image_display()  # Ensure the image is updated in the UI

        root.title(f"Reviewing: {img_files[img_index]} ({img_index+1}/{len(img_files)})")

        # Load existing classification if available
        filename = img_files[img_index]
        if filename in annotations:
            print(f"📋 Loading existing classifications for {filename}")
            for i, key in enumerate(classification_keys):
                classification[key] = annotations[filename][i]
        else:
            print(f"🆕 No existing classification for {filename}, setting defaults.")
            reset_classifications()

        update_status()
        root.update()  # Force UI refresh
    except Exception as e:
        print(f"❌ Error loading image {img_files[img_index]}: {str(e)}")

# Update image display
def update_image_display():
    global img_tk
    if current_img:
        try:
            img_frame_width = img_frame.winfo_width()
            img_frame_height = img_frame.winfo_height() - STATUS_HEIGHT  # Leave space for status

            print(f"🔍 Resizing image to fit within ({img_frame_width}, {img_frame_height})")

            resized_img = current_img.resize((img_frame_width, img_frame_height), Image.LANCZOS)

            img_tk = ImageTk.PhotoImage(resized_img)
            img_label.config(image=img_tk)
            img_label.image = img_tk  # Keep reference to avoid garbage collection

            print(f"✅ Image updated in UI")
        except Exception as e:
            print(f"❌ Error updating image display: {str(e)}")

# Move forward
def next_image():
    global img_index
    print(f"➡️ Moving to next image from {img_index}")
    save_classifications()
    img_index = (img_index + 1) % len(img_files)  # Loop at the end
    load_image()

# Move backward
def prev_image():
    global img_index
    print(f"⬅️ Moving to previous image from {img_index}")
    save_classifications()
    img_index = (img_index - 1) % len(img_files)  # Loop at the beginning
    load_image()

# Reset classification to "no"
def reset_classifications():
    for key in classification.keys():
        classification[key] = "no"
    print("🔄 Reset classifications to default.")

# Update status label
def update_status():
    try:
        status_text = " | ".join([f"{key}: {value}" for key, value in classification.items()])
        status_label.config(text=status_text)
        print(f"ℹ️ Status updated: {status_text}")
    except Exception as e:
        print("❌ Error updating status:", str(e))

# Save classification to CSV
def save_classifications():
    try:
        filename = img_files[img_index]
        annotations[filename] = [classification[key] for key in classification_keys]

        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename"] + classification_keys)
            for key in sorted(annotations):  
                writer.writerow([key] + annotations[key])

        print(f"💾 Saved classifications for {filename}")
    except Exception as e:
        print("❌ Error saving classifications:", str(e))

def toggle_classification(label):
    try:
        classification[label] = "yes" if classification[label] == "no" else "no"
        update_status()
        print(f"🔄 Toggled {label} -> {classification[label]}")
    except Exception as e:
        print(f"❌ Error toggling {label}:", str(e))


# Keyboard bindings
for i, key in enumerate(classification_keys, start=1):
    root.bind(str(i), lambda e, k=key: toggle_classification(k))

root.bind("<Right>", lambda e: next_image())  # Move to next image
root.bind("<Left>", lambda e: prev_image())   # Move to previous image

# Start application
print("🚀 Starting image review app...")
load_image()
root.mainloop()
