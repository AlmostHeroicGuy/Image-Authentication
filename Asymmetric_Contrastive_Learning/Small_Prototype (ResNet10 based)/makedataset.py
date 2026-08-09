from datasets import load_dataset
import os

dataset = load_dataset("zh-plus/tiny-imagenet")

os.makedirs("TinyImageNet/train", exist_ok=True)
os.makedirs("TinyImageNet/test", exist_ok=True)

# for i, ex in enumerate(dataset["train"]):
#     ex["image"].convert("RGB").save(
#         f"TinyImageNet/train/{i:07d}.png"
#     )

for i, ex in enumerate(dataset["valid"]):
    ex["image"].convert("RGB").save(
        f"TinyImageNet/test/{i:05d}.png"
    )


print("Done!")