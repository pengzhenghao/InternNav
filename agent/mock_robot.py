from flask import Flask, jsonify, request
import base64
import os
import logging
from PIL import Image, ImageDraw
import io

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockRobot")

current_instruction = "WAIT"

def create_dummy_image(text="Robot View"):
    """Generates a dummy image with text"""
    img = Image.new('RGB', (640, 480), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), text, fill=(255,255,0))
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route("/ui_data")
def ui_data():
    """Simulates the robot's status endpoint"""
    return jsonify({
        "image": create_dummy_image(f"Seeing: Office\nInstr: {current_instruction}"),
        "instruction": current_instruction,
        "logs": ["System normal"]
    })

@app.route("/set_instruction", methods=['POST'])
def set_instruction():
    """Simulates the robot receiving a new command"""
    global current_instruction
    data = request.json
    new_instr = data.get('instruction', '')
    logger.info(f"Robot received new instruction: {new_instr}")
    current_instruction = new_instr
    return jsonify({"status": "success"})

if __name__ == "__main__":
    print("Starting Mock Robot Server on port 5802...")
    app.run(host='0.0.0.0', port=5802)
