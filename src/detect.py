from ultralytics import YOLO


MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "data/video/visdrone_test.mp4"


def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=VIDEO_PATH,
        device=0,
        conf=0.25,
        save=True
    )

    print("Detection complete")


if __name__ == "__main__":
    main()