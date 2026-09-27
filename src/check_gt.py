from pathlib import Path
import cv2
import pandas as pd
import subprocess


# =============================
# 1. 경로 설정
# =============================

SEQUENCE_NAME = "uav0000137_00458_v"

# 원본 이미지 프레임 폴더
IMAGE_DIR = Path(
    "data/video/VisDrone2019-MOT-val/"
    f"sequences/{SEQUENCE_NAME}"
)

# GT annotation 파일
GT_PATH = Path(
    "data/video/VisDrone2019-MOT-val/"
    f"annotations/{SEQUENCE_NAME}.txt"
)

# GT 박스를 그린 이미지 저장 폴더
GT_FRAME_DIR = Path("data/video/gt_frames")

# 최종 GT 영상
OUTPUT_VIDEO = Path("data/video/gt_visdrone.mp4")

FPS = 30


# =============================
# 2. VisDrone 클래스 이름
# =============================

CLASS_NAMES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


# =============================
# 3. GT annotation 읽기
# =============================

def load_gt():

    # VisDrone annotation 열 이름
    columns = [
        "frame", "target_id", "x", "y", "w", "h",
        "score", "category", "truncation", "occlusion"
    ]

    # txt 파일 읽기
    gt = pd.read_csv(GT_PATH, header=None, names=columns)

    # score가 1인 평가 대상 객체만 사용
    gt = gt[gt["score"] == 1]

    return gt


# =============================
# 4. 이미지에 GT 박스 그리기
# =============================

def draw_gt_frames(gt):

    # 저장 폴더 생성
    GT_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # JPG 파일 목록
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    print("전체 프레임 수:", len(image_files))

    # 이미지 한 장씩 처리
    for image_path in image_files:

        # 0000001.jpg -> 1
        frame_number = int(image_path.stem)

        # 이미지 읽기
        frame = cv2.imread(str(image_path))

        # 현재 프레임의 GT만 가져오기
        frame_gt = gt[gt["frame"] == frame_number]

        # 현재 프레임 객체 하나씩 처리
        for _, row in frame_gt.iterrows():

            # 객체 ID / 클래스
            target_id = int(row["target_id"])
            category = int(row["category"])

            # Bounding Box 좌표
            x1 = int(row["x"])
            y1 = int(row["y"])
            x2 = x1 + int(row["w"])
            y2 = y1 + int(row["h"])

            # 클래스 이름
            class_name = CLASS_NAMES.get(category, "unknown")

            # Bounding Box 그리기
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ID + 클래스 표시
            label = f"GT ID:{target_id} {class_name}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # GT 박스를 그린 이미지 저장
        output_path = GT_FRAME_DIR / image_path.name
        cv2.imwrite(str(output_path), frame)

        print(f"\r처리 중: {frame_number}", end="")

    print()


# =============================
# 5. FFmpeg로 영상 생성
# =============================

def make_video():

    # FFmpeg 명령
    command = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", str(GT_FRAME_DIR / "%07d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(OUTPUT_VIDEO)
    ]

    print("영상 생성 중...")

    # FFmpeg 실행
    subprocess.run(command, check=True)

    print("영상 생성 완료:", OUTPUT_VIDEO)


# =============================
# 6. 전체 실행
# =============================

def main():

    # GT 불러오기
    gt = load_gt()

    print("GT 객체 수:", len(gt))

    # 각 이미지에 GT 박스 그리기
    draw_gt_frames(gt)

    # 이미지 -> MP4 변환
    make_video()


if __name__ == "__main__":
    main()