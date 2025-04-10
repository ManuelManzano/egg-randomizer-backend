from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import random
import base64

app = Flask(__name__)
CORS(app)

# ✅ Health check route for Render to detect open port
@app.route('/')
def health_check():
    return "Egg Randomizer Backend is Live!", 200

# ✅ Load YOLO model
model = YOLO("runs/detect/train10/weights/best.pt")
TEST_IMAGE_DIR = "datasets/EggRandomizerDataset/test_images"

@app.route('/api/egg-seed', methods=['GET'])
def generate_egg_seed():
    try:
        images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            return jsonify({"error": "No test images found"}), 404

        selected_image = random.choice(images)
        image_path = os.path.join(TEST_IMAGE_DIR, selected_image)

        image = cv2.imread(image_path)
        results = model(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        egg_colors = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            patch = image[cy - 5:cy + 5, cx - 5:cx + 5]
            if patch.size > 0:
                avg_color = patch.mean(axis=(0, 1))
                rgb = avg_color[::-1]
                egg_colors.append(rgb.tolist())

        egg_count = len(egg_colors)
        chicken_index = np.random.randint(1, 10)
        rgb_sum = np.sum(egg_colors)
        raw_seed = rgb_sum + egg_count * 10 + chicken_index
        normalized_seed = (raw_seed % 10000) / 10000.0

        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        base64_image = f"data:image/jpeg;base64,{encoded_image}"

        return jsonify({
            "seed": normalized_seed,
            "egg_count": egg_count,
            "egg_colors": egg_colors,
            "image": base64_image
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Proper port handling for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("Starting server on port:", port)
    app.run(host='0.0.0.0', port=port, debug=False)
