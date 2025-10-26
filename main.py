import io
import json
import time
import random
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Tuple
from collections import Counter
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Pose metadata
pose_library = {
    "DownDog": {
        "level": "Beginner",
        "focus": ["Flexibility", "Strength", "Relaxation"],
        "time": 120,
        "type": "warm-up",
        "description": "Hands and feet on the ground, hips raised, forming an inverted V shape.",
        "image": "yogamind/backend/images/DownDog.jpg"
    },
    "Warrior2": {
        "level": "Intermediate",
        "focus": ["Strength", "Flexibility", "Relaxation"],
        "time": 90,
        "type": "warm-up",
        "description": "Legs in a lunge, arms extended parallel to the ground, gazing over front hand.",
        "image": "yogamind/backend/images/Warrior2.jpg"
    },
    "Lotus": {
        "level": "Advanced",
        "focus": ["Flexibility", "Relaxation", "Strength"],
        "time": 120,
        "type": "cool-down",
        "description": "Seated with legs crossed, each foot on opposite thigh, hands resting on knees.",
        "image": "yogamind/backend/images/Lotus.jpg"
    },
    "Plank": {
        "level": "Beginner",
        "focus": ["Strength", "Relaxation"],
        "time": 60,
        "type": "main",
        "description": "Body straight, supported on forearms and toes, core engaged.",
        "image": "yogamind/backend/images/Plank.jpg"
    },
    "Tree": {
        "level": "Intermediate",
        "focus": ["Flexibility", "Relaxation", "Strength"],
        "time": 60,
        "type": "main",
        "description": "Standing on one leg, other foot placed on inner thigh, hands in prayer position.",
        "image": "yogamind/backend/images/Tree.jpg"
    },
}

# Load TorchScript model
try:
    yoga_model = torch.jit.load("yoga_model_scripted.pt", map_location=device)
    yoga_model.eval()
    logger.info("✅ TorchScript model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    yoga_model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# FastAPI app setup
app = FastAPI(title="Yoga Pose Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
def generate_routine(level: str, goal: str, duration: int) -> List[Tuple[str, int]]:
    total_seconds = duration * 60
    goal_map = {
        "Weight Loss": "Strength",
        "Back Pain Relief": "Flexibility",
    }
    focus = goal_map.get(goal, goal)
    all_poses = {p: v for p, v in pose_library.items() if v["level"] == level and focus in v["focus"]}
    if not all_poses:
        return []
    routine, current_time = [], 0
    while current_time < total_seconds:
        pose = random.choice(list(all_poses.keys()))
        time_spent = all_poses[pose]["time"]
        if current_time + time_spent > total_seconds:
            break
        routine.append((pose, time_spent))
        current_time += time_spent
    return routine


@app.post("/generate_routine")
async def generate_yoga_routine(level: str, goal: str, duration: int):
    if not level or not goal or not duration:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    routine = generate_routine(level, goal, duration)
    if not routine:
        raise HTTPException(status_code=404, detail="No suitable poses found")

    response = {
        "routine": [
            {
                "pose": pose,
                "time_seconds": time,
                "description": pose_library[pose]["description"],
                "image": pose_library[pose]["image"]
            } for pose, time in routine
        ]
    }
    return JSONResponse(content=response)


@app.post("/detect_pose_image")
async def detect_pose_image(file: UploadFile = File(...)):
    if not yoga_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = yoga_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0][pred_idx].item()

        # If your class labels are known:
        classes = list(pose_library.keys())
        pose_name = classes[pred_idx % len(classes)]

        return {
            "pose": pose_name,
            "confidence": confidence,
            "description": pose_library.get(pose_name, {}).get("description", "Unknown pose")
        }

    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Image prediction failed")


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

