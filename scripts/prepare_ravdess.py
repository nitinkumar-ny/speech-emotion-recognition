import os
import shutil

SOURCE_DIR = "Audio_Speech_Actors_01-24"
TARGET_DIR = "data/ravdess"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}
print("TARGET_DIR : ", TARGET_DIR)
os.makedirs(TARGET_DIR, exist_ok=True)

for emotion in EMOTION_MAP.values():
    os.makedirs(os.path.join(TARGET_DIR, emotion), exist_ok=True)

for actor in os.listdir(SOURCE_DIR):
    actor_path = os.path.join(SOURCE_DIR, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        emotion_code = file.split("-")[2]

        if emotion_code in EMOTION_MAP:
            emotion = EMOTION_MAP[emotion_code]
            src = os.path.join(actor_path, file)
            dst = os.path.join(TARGET_DIR, emotion, file)
            shutil.copy(src, dst)

print("RAVDESS dataset prepared successfully!")
