import io
import random
import pyttsx3
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List, Tuple
import logging
from fastai.vision.all import *
import sys
import pathlib
import torch
import time
from PIL import Image
from collections import Counter


if sys.platform == "win32":
    class PosixPath(pathlib.WindowsPath):
        def __new__(cls, *args, **kwargs):
            return pathlib.WindowsPath(*args, **kwargs)
    pathlib.PosixPath = PosixPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

pose_library = {
    "DownDog": {
        "level": "Beginner",
        "focus": ["Flexibility", "Strength","Relaxation"],
        "time": 120,
        "type": "warm-up",
        "description": "Hands and feet on the ground, hips raised, forming an inverted V shape.",
        "image": "yogamind\backend\images\DownDog.jpg"
    },
    "Warrior2": {
        "level": "Intermediate",
        "focus": ["Strength", "Flexibility","Relaxation"],
        "time": 90,
        "type": "warm-up",
        "description": "Legs in a lunge, arms extended parallel to the ground, gazing over front hand.",
        "image": "yogamind\backend\images\Warrior2.jpg"
    },
    "Lotus": {
        "level": "Advanced",
        "focus": ["Flexibility", "Relaxation","Strength"],
        "time": 120,
        "type": "cool-down",
        "description": "Seated with legs crossed, each foot on opposite thigh, hands resting on knees.",
        "image": "yogamind\backend\images\Lotus.jpg"
    },
    "Plank": {
        "level": "Beginner",
        "focus": ["Strength","Relaxation"],
        "time": 60,
        "type": "main",
        "description": "Body straight, supported on forearms and toes, core engaged.",
        "image": "yogamind\backend\images\Plank.jpg"
    },
    "Staff": {
        "level": "Intermediate",
        "focus": ["Flexibility"],
        "time": 90,
        "type": "main",
        "description": "Seated with legs extended forward, back straight, hands by hips.",
        "image": "yogamind\backend\images\Staff.jpg"
    },
    "Diamond": {
        "level": "Beginner",
        "focus": ["Stength","Relaxation"],
        "time": 120,
        "type": "cool-down",
        "description": "Seated with soles of feet together, knees bent outward, hands holding feet.",
        "image": "yogamind\backend\images\Diamond.jpg"
    },
    "Tree": {
        "level": "Intermediate",
        "focus": ["Flexibility","Relaxation","Strength"],
        "time": 60,
        "type": "main",
        "description": "Standing on one leg, other foot placed on inner thigh, hands in prayer position.",
        "image": "yogamind\backend\images\Tree.jpg"
    },
    "Goddess": {
        "level": "Intermediate",
        "focus": ["Strength", "Flexibility"],
        "time": 90,
        "type": "main",
        "description": "Wide stance, knees bent, arms bent at elbows, palms up.",
        "image": "yogamind\backend\images\Goddess.jpg"
    },
    "Staff": {
        "level": "Advanced",
        "focus": ["Flexibility","Strength"],
        "time": 90,
        "type": "main",
        "description": "Seated with legs extended forward, back straight, hands by hips.",
        "image": "yogamind\backend\images\Staff.jpg"
        }
}


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
    warm_up_poses = {p: v for p, v in all_poses.items() if v["type"] == "warm-up"}
    main_poses = {p: v for p, v in all_poses.items() if v["type"] == "main"}
    cool_down_poses = {p: v for p, v in all_poses.items() if v["type"] == "cool-down"}
    routine = []
    current_time = 0
    if warm_up_poses:
        pose = random.choice(list(warm_up_poses.keys()))
        time = warm_up_poses[pose]["time"]
        if current_time + time <= total_seconds:
            routine.append((pose, time))
            current_time += time
    while main_poses and current_time < total_seconds * 0.8:
        pose = random.choice(list(main_poses.keys()))
        time = main_poses[pose]["time"]
        if current_time + time > total_seconds:
            continue
        routine.append((pose, time))
        current_time += time
    if cool_down_poses:
        pose = random.choice(list(cool_down_poses.keys()))
        time = cool_down_poses[pose]["time"]
        if current_time + time <= total_seconds:
            routine.append((pose, time))
            current_time += time
    while current_time < total_seconds and main_poses:
        pose = random.choice(list(main_poses.keys()))
        time = main_poses[pose]["time"]
        if current_time + time > total_seconds:
            continue
        routine.append((pose, time))
        current_time += time
    return routine

def aggregate_routine(routine: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    pose_times = {}
    for pose, time in routine:
        pose_times[pose] = pose_times.get(pose, 0) + time
    return [(pose, time) for pose, time in pose_times.items()]

yoga_model = None
try:
    yoga_model = load_learner("yoga_model_windows.pkl")
    yoga_model.model.to(device)
    yoga_model.model.eval()
    logger.info("FastAI model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FastAI model: {e}")
    yoga_model = None

app = FastAPI(title="Yoga Pose Detection and Routine API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate_routine")
async def generate_yoga_routine(level: str, goal: str, duration: int):
    if level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid level")
    if goal not in ["Relaxation", "Flexibility", "Strength", "Weight Loss", "Back Pain Relief"]:
        raise HTTPException(status_code=400, detail="Invalid goal")
    if duration not in [10, 20, 30]:
        raise HTTPException(status_code=400, detail="Invalid duration")
    
    routine = generate_routine(level, goal, duration)
    if not routine:
        raise HTTPException(status_code=404, detail="No poses found for your criteria")
    
    aggregated = aggregate_routine(routine)
    response = {
        "routine": [
            {
                "pose": pose,
                "time_seconds": time,
                "description": pose_library[pose]["description"],
                "image": pose_library[pose]["image"]
            } for pose, time in aggregated
        ]
    }
    return JSONResponse(content=response)


@app.post("/detect_pose_image")
async def detect_pose_image(file: UploadFile = File(...)):
    if not yoga_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    contents = await file.read()
    img = PILImage.create(io.BytesIO(contents))
    
    start_time = time.time()
    try:
        test_dl = yoga_model.dls.test_dl([img])
        with torch.no_grad():
            preds = yoga_model.get_preds(dl=test_dl, reorder=False)
            pred_idx = preds[0].argmax(dim=1).item()
            probs = torch.softmax(preds[0], dim=1)
            confidence = probs[0][pred_idx].item()
            pred = yoga_model.dls.vocab[pred_idx]
        logger.info(f"Image prediction time: {time.time() - start_time:.2f}s")
        return {
            "pose": pred,
            "confidence": confidence,
            "description": pose_library.get(pred, {}).get("description", "Unknown pose")
        }
    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Image prediction failed")

@app.post("/detect_pose_video")
async def detect_pose_video(file: UploadFile = File(...)):
    if not yoga_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    contents = await file.read()
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(contents)
    
    start_time = time.time()
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        raise HTTPException(status_code=400, detail="Failed to open video file")
    
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    predictions = []
    frame_count = 0
    batch_images = []
    batch_frames = []
    batch_size = 8  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 != 0:  # Sample every 10th frame
            out.write(frame)
            frame_count += 1
            continue
        img = PILImage.create(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        batch_images.append(img)
        batch_frames.append(frame)
        frame_count += 1
        
        # Process batch when full or at end
        if len(batch_images) == batch_size or (not ret and batch_images):
            try:
                start_batch = time.time()
                test_dl = yoga_model.dls.test_dl(batch_images)
                with torch.no_grad():
                    preds = yoga_model.get_preds(dl=test_dl, reorder=False)
                    probs = torch.softmax(preds[0], dim=1)
                    pred_indices = preds[0].argmax(dim=1)
                for i, (img, frame) in enumerate(zip(batch_images, batch_frames)):
                    pred_idx = pred_indices[i].item()
                    confidence = probs[i][pred_idx].item()
                    pred = yoga_model.dls.vocab[pred_idx]
                    pred_str = f"{pred} ({confidence:.2f})"
                    cv2.putText(frame, pred_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    out.write(frame)
                    predictions.append({
                        "frame": frame_count - len(batch_images) + i,
                        "pose": pred,
                        "confidence": confidence
                    })
                logger.info(f"Batch prediction time: {time.time() - start_batch:.2f}s for {len(batch_images)} frames")
            except Exception as e:
                logger.error(f"Batch prediction failed at frame {frame_count}: {e}")
            batch_images = []
            batch_frames = []
    
    cap.release()
    out.release()
    
    
    try:
        with open("video_predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved {len(predictions)} predictions to video_predictions.json")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
    
    logger.info(f"Video processing time: {time.time() - start_time:.2f}s for {len(predictions)} predictions")
    
    
    if not predictions:
        raise HTTPException(status_code=404, detail="No poses detected in the video")
    
    
    pose_counts = Counter(pred["pose"] for pred in predictions)
    most_common_pose, _ = pose_counts.most_common(1)[0]
    
    
    confidences = [pred["confidence"] for pred in predictions if pred["pose"] == most_common_pose]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    
    return JSONResponse(content={
        "pose": most_common_pose,
        "confidence": avg_confidence,
        "description": pose_library.get(most_common_pose, {}).get("description", "Unknown pose")
    })


@app.post("/voice_guidance")
async def get_voice_guidance(level: str, goal: str, duration: int):
    routine = generate_routine(level, goal, duration)
    if not routine:
        raise HTTPException(status_code=404, detail="No routine generated")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    aggregated = aggregate_routine(routine)
    
    audio_buffer = io.BytesIO()
    for i, (pose, t) in enumerate(aggregated, 1):
        mins = t // 60
        secs = t % 60
        time_str = f"{mins} minute{'s' if mins != 1 else ''}" if secs == 0 else f"{mins} minute{'s' if mins != 1 else ''} and {secs} seconds"
        engine.say(f"Step {i}. {pose} for {time_str}. {pose_library[pose]['description']}")
    engine.save_to_file("", audio_buffer)
    engine.runAndWait()
    audio_buffer.seek(0)
    return JSONResponse(content={"message": "Voice guidance generated - integrate audio streaming if needed"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
