import torch
import matplotlib.pyplot as plt

img = plt.imread("../data/images/bus.jpg")
plt.figure("Image")  # 图像窗口名称
plt.show()

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom


# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.