from pathlib import Path

import motmetrics as mm
import pandas as pd


# =========================================
# 1. 파일 경로 설정
# =========================================

GT_PATH = Path(
    "data/video/VisDrone2019-MOT-val/"
    "annotations/uav0000137_00458_v.txt"
)

PRED_PATH = Path("outputs/track_results.csv")

# Bounding Box가 50% 이상 겹치면 같은 객체 후보로 판단
IOU_THRESHOLD = 0.5


# =========================================
# 2. YOLO 클래스 -> VisDrone 클래스 매핑
# =========================================

YOLO_TO_VISDRONE = {
    0: [1, 2],      # person -> pedestrian, people
    1: [3],         # bicycle -> bicycle
    2: [4, 5],      # car -> car, van
    3: [10],        # motorcycle -> motor
}

# YOLO와 비교할 VisDrone 클래스
VISDRONE_CLASSES = [1, 2, 3, 4, 5, 10]

# 이번 평가에서 사용하는 YOLO 클래스
YOLO_CLASSES = [0, 1, 2, 3]


# =========================================
# 3. Ground Truth 읽기
# =========================================

def load_gt():

    # VisDrone annotation 열 이름
    columns = [
        "frame", "target_id", "x", "y", "w", "h",
        "score", "category", "truncation", "occlusion"
    ]

    # txt 파일 읽기
    gt = pd.read_csv(GT_PATH, header=None, names=columns)

    # score=1이고 YOLO와 매칭 가능한 클래스만 평가
    gt = gt[
        (gt["score"] == 1)
        & (gt["category"].isin(VISDRONE_CLASSES))
    ].copy()

    return gt


# =========================================
# 4. ByteTrack 예측 결과 읽기
# =========================================

def load_predictions():

    # track.py에서 저장한 CSV 읽기
    pred = pd.read_csv(PRED_PATH)

    # class는 Python 예약어라 이름 변경
    pred = pred.rename(columns={"class": "yolo_class"})

    # x1, y1, x2, y2 -> width, height 계산
    pred["w"] = pred["x2"] - pred["x1"]
    pred["h"] = pred["y2"] - pred["y1"]

    # 이번 평가에 필요한 YOLO 클래스만 사용
    pred = pred[
        pred["yolo_class"].isin(YOLO_CLASSES)
    ].copy()

    return pred


# =========================================
# 5. Tracking 성능 평가
# =========================================

def evaluate(gt, pred):

    # MOT 평가 결과 저장 객체
    accumulator = mm.MOTAccumulator(auto_id=True)

    # GT와 예측에 존재하는 모든 프레임 번호
    all_frames = sorted(
        set(gt["frame"]) | set(pred["frame"])
    )

    # 프레임 하나씩 비교
    for frame in all_frames:

        # 현재 프레임 GT / 예측
        gt_frame = gt[gt["frame"] == frame]
        pred_frame = pred[pred["frame"] == frame]

        # GT / ByteTrack 객체 ID
        gt_ids = gt_frame["target_id"].tolist()
        pred_ids = pred_frame["track_id"].tolist()

        # Bounding Box
        gt_boxes = gt_frame[
            ["x", "y", "w", "h"]
        ].to_numpy()

        pred_boxes = pred_frame[
            ["x1", "y1", "w", "h"]
        ].to_numpy()

        # IoU 0.5 이상인 객체만 매칭 가능
        distance_matrix = mm.distances.iou_matrix(
            gt_boxes,
            pred_boxes,
            max_iou=1 - IOU_THRESHOLD
        )

        # 클래스가 다르면 매칭 금지
        for gt_index, gt_row in enumerate(gt_frame.itertuples()):
            for pred_index, pred_row in enumerate(pred_frame.itertuples()):

                visdrone_class = int(gt_row.category)
                yolo_class = int(pred_row.yolo_class)

                # YOLO 클래스와 대응되는 VisDrone 클래스
                allowed_classes = YOLO_TO_VISDRONE.get(yolo_class, [])

                # 클래스가 다르면 매칭 불가능
                if visdrone_class not in allowed_classes:
                    distance_matrix[gt_index, pred_index] = float("nan")

        # 현재 프레임 평가 결과 저장
        accumulator.update(
            gt_ids,
            pred_ids,
            distance_matrix
        )

    return accumulator


# =========================================
# 6. 평가 결과 출력
# =========================================

def print_result(accumulator):

    # motmetrics 평가 도구
    metrics = mm.metrics.create()

    # 사용할 평가 지표
    summary = metrics.compute(
        accumulator,
        metrics=[
            "num_frames",            # 평가한 프레임 수
            "num_objects",           # GT 객체 수
            "num_matches",           # 정상 매칭 수
            "num_misses",            # 놓친 객체 수
            "num_false_positives",   # 잘못 탐지한 객체 수
            "num_switches",          # ID Switch 횟수
            "mota",                  # MOT 종합 성능
            "idf1",                  # ID 유지 성능
            "precision",             # 정밀도
            "recall",                # 재현율
        ],
        name="YOLO26n + ByteTrack"
    )

    print("\n=== Tracking Evaluation ===")

    print(
        mm.io.render_summary(
            summary,
            formatters={
                "mota": "{:.3f}".format,
                "idf1": "{:.3f}".format,
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
            }
        )
    )


# =========================================
# 7. 실행
# =========================================

def main():

    # GT / 예측 결과 불러오기
    gt = load_gt()
    pred = load_predictions()

    print("평가용 GT 객체 수:", len(gt))
    print("평가용 예측 객체 수:", len(pred))

    # GT와 예측 비교 후 결과 출력
    accumulator = evaluate(gt, pred)
    print_result(accumulator)


if __name__ == "__main__":
    main()