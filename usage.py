import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from IPython.display import display
import ipywidgets as widgets

 

CONFIDENCE_THRESHOLD = 0.75
NMS_IOU_THRESHOLD    = 0.75

 

model         = YOLO("checkbox.pt")
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

 

def callback(frame: np.ndarray, frame_id: int) -> np.ndarray:
    kernel    = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    sharpened = cv2.filter2D(frame, -1, kernel)

    results = model(
        sharpened,
        conf=CONFIDENCE_THRESHOLD,
        iou=NMS_IOU_THRESHOLD
    )[0]

    detections = sv.Detections.from_ultralytics(results)
    labels = [
        f"{results.names[cid]} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    out = box_annotator.annotate(sharpened.copy(), detections=detections)
    out = label_annotator.annotate(out, detections=detections, labels=labels)
    return out

 

path_to_tiff = "/PATH-TO-TIFF"
success, pages = cv2.imreadmulti(path_to_tiff, flags=cv2.IMREAD_COLOR)

if not success:
    raise FileNotFoundError(f"Could not load TIFF file: {path_to_tiff}")
widgets_list = []
for idx, page in enumerate(pages):
    annotated = callback(page, idx)
    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpeg', rgb_frame)
    img_w = widgets.Image(value=buffer.tobytes(), format='jpeg')
    widgets_list.append(img_w)
cols = 2
grid  = widgets.GridBox(
    widgets_list,
    layout=widgets.Layout(
        grid_template_columns=f"repeat({cols}, 1fr)",
        grid_gap="8px"
    )
)
display(grid)
