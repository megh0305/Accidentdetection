import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

prev_positions = []

def overlap(box1, box2):
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2

    return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    boxes = []

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car","truck","bus","motorcycle"]:
                boxes.append((x1,y1,x2,y2))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    accident = False

    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if overlap(boxes[i], boxes[j]):
                accident = True

    if accident:
        cv2.putText(frame,"ACCIDENT DETECTED!",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),3)

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
