from ultralytics import YOLO


def detect_with_yolo(fname: str):
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model.predict(
        source="/mnt/e/wfs/yz/ORG-000-ffb2665685ad48c893ff8a3359233539d5815489508d6dce3be80a87fdceea8d.jpg", show=True, conf=0.25)