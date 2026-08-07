import os
import random
import shutil

random.seed(42)

train_dir = "dataset/train"
val_dir = "dataset/val"

classes = ["Forest", "NonForest"]

for cls in classes:
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    images = os.listdir(os.path.join(train_dir, cls))
    random.shuffle(images)

    val_count = int(len(images) * 0.2)

    for img in images[:val_count]:
        src = os.path.join(train_dir, cls, img)
        dst = os.path.join(val_dir, cls, img)
        shutil.move(src, dst)

print("Dataset split completed!")