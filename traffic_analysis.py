import cv2
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = os.path.join("runs", "train", "exp", "weights", "best.pt")
VIDEO_PATH = r"counting_cars.mp4"
OUTPUT_PATH = r"result.mp4"

POLYGONS = [
    np.array([[1061, 193], [1061, 471], [757, 469], [757, 190]]),
    np.array([[1058, 489], [1063, 773], [757, 775], [752, 494]]),
    np.array([[427, 786], [430, 906], [312, 903], [310, 786]]),
    np.array([[430, 929], [435, 1069], [317, 1069], [315, 931]]),
    np.array([[1546, 37], [1554, 172], [1692, 172], [1689, 34]])
]


def main():
    model = YOLO(MODEL_PATH)
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    tracker = sv.ByteTrack()
    colors = sv.ColorPalette.DEFAULT

    #create zones and counter variable
    zones = []
    zone_annotators = []

    #create id for each car
    zone_counted_ids = [set() for _ in POLYGONS]

    for i, polygon in enumerate(POLYGONS):
        # Logic zone
        zone = sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=[sv.Position.CENTER]
        )
        zones.append(zone)

        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(i),
            thickness=2,
            text_scale=0,
            text_thickness=0
        )
        zone_annotators.append(zone_annotator)

    # Car drawing tool
    trace_annotator = sv.TraceAnnotator(trace_length=30, thickness=2)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)

    frame_generator = sv.get_video_frames_generator(VIDEO_PATH)

    print(f"Video in process: {VIDEO_PATH}")

    with sv.VideoSink(target_path=OUTPUT_PATH, video_info=video_info) as sink:

        for frame in frame_generator:
            # 1. Detect
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # 2. Track
            detections = tracker.update_with_detections(detections)

            # 3. Basic annotation (Drawing the car, tail, ID labels)
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # 4. Logic process
            for i, zone in enumerate(zones):
                is_in_zone = zone.trigger(detections=detections)
                ids_in_zone = detections.tracker_id[is_in_zone]
                zone_counted_ids[i].update(ids_in_zone)
                current_total_count = len(zone_counted_ids[i])
                annotated_frame = zone_annotators[i].annotate(scene=annotated_frame)
                text_anchor = POLYGONS[i][0]
                x, y = int(text_anchor[0]), int(text_anchor[1])
                text = f"Zone {i + 1}: {current_total_count}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(annotated_frame, (x, y - h - 10), (x + w + 10, y + 5), colors.by_idx(i).as_bgr(), -1)

                cv2.putText(annotated_frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            sink.write_frame(annotated_frame)

            cv2.imshow("Cumulative Counting", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    print(f"{'Result'}")
    for i, count_set in enumerate(zone_counted_ids):
        print(f" Zone {i + 1}: {len(count_set)} vehicle(s)")
    print(f"Save at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()