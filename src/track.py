from ultralytics import YOLO
from pathlib import Path
import cv2
import csv
import subprocess


MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "data/video/visdrone_test.mp4"
TRACKER_PATH = "bytetrack.yaml"

OUTPUT_CSV = Path("outputs/track_results.csv")
FRAME_DIR = Path("outputs/track_frames")
OUTPUT_VIDEO = Path("outputs/track_mapped.mp4")


# YOLO 클래스 -> VisDrone 클래스명 매핑
CLASS_MAPPING = {
    0: "pedestrian/people",
    1: "bicycle",
    2: "car/van",
    3: "motor",
}


def main():

    # 결과 폴더 생성
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # YOLO 모델 불러오기
    model = YOLO(MODEL_PATH)

    # YOLO + ByteTrack 실행
    results = model.track(
        source=VIDEO_PATH,
        tracker=TRACKER_PATH,
        device=0,
        conf=0.25,
        persist=True,
        stream=True,
        save=False
    )

    # CSV 파일 생성
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame", "track_id", "class", "mapped_class",
            "confidence", "x1", "y1", "x2", "y2"
        ])

        # 프레임 하나씩 처리
        for frame_idx, result in enumerate(results, start=1):

            frame = result.orig_img.copy()
            boxes = result.boxes

            # Track ID가 있는 객체만 처리
            if boxes.id is not None:

                for box, track_id in zip(boxes, boxes.id):

                    # YOLO 클래스 / 신뢰도
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Bounding Box 좌표
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # VisDrone 기준 클래스명으로 변환
                    mapped_class = CLASS_MAPPING.get(
                        cls_id,
                        model.names[cls_id]
                    )

                    # CSV 저장
                    writer.writerow([
                        frame_idx,
                        int(track_id),
                        cls_id,
                        mapped_class,
                        confidence,
                        x1, y1, x2, y2
                    ])

                    # Bounding Box 그리기
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # ID + 클래스 + confidence 표시
                    label = f"ID:{int(track_id)} {mapped_class} {confidence:.2f}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

            # 프레임 이미지 저장
            frame_path = FRAME_DIR / f"{frame_idx:07d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            print(f"\rframe {frame_idx}", end="")

    print()

    # FFmpeg로 프레임 이미지 -> MP4 변환
    command = [
        "ffmpeg",
        "-y",
        "-framerate", "30",
        "-i", str(FRAME_DIR / "%07d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(OUTPUT_VIDEO)
    ]

    subprocess.run(command, check=True)

    print("Tracking CSV 저장:", OUTPUT_CSV)
    print("Tracking 영상 저장:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()