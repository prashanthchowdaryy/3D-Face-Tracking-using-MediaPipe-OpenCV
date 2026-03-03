import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False
points = []

def detect_shape(points, w, h):
    if len(points) < 40:
        return None

    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], 255, 2)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 3:
        return ("triangle", approx)

    elif len(approx) == 4:
        x, y, w_box, h_box = cv2.boundingRect(approx)
        aspect = w_box / float(h_box)

        if 0.9 <= aspect <= 1.1:
            return ("square", (x, y, w_box))
        else:
            return ("rectangle", (x, y, w_box, h_box))

    elif len(approx) > 4:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        return ("circle", (int(x), int(y), int(r)))

    return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    canvas = cv2.resize(canvas, (w, h))

    preview = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            mx, my = int(lm[12].x * w), int(lm[12].y * h)

            index_up = iy < int(lm[6].y * h)
            middle_up = my < int(lm[10].y * h)

            if index_up and not middle_up:
                drawing = True
                points.append((ix, iy))

                if len(points) > 1:
                    cv2.line(preview, points[-2], points[-1], (0, 0, 255), 2)

            else:
                if drawing:
                    shape = detect_shape(points, w, h)

                    if shape:
                        kind, data = shape

                        if kind == "circle":
                            cv2.circle(canvas, data[:2], data[2], (0, 255, 0), 4)

                        elif kind == "rectangle":
                            x, y, bw, bh = data
                            cv2.rectangle(
                                canvas, (x, y), (x + bw, y + bh), (255, 0, 0), 4
                            )

                        elif kind == "square":
                            x, y, s = data
                            cv2.rectangle(
                                canvas, (x, y), (x + s, y + s), (255, 255, 0), 4
                            )

                        elif kind == "triangle":
                            cv2.drawContours(
                                canvas, [data], -1, (255, 0, 255), 4
                            )

                    else:
                        for i in range(len(points) - 1):
                            cv2.line(
                                canvas,
                                points[i],
                                points[i + 1],
                                (0, 0, 255),
                                3
                            )

                points = []
                drawing = False

            mp_draw.draw_landmarks(preview, hand, mp_hands.HAND_CONNECTIONS)

    frame = cv2.addWeighted(preview, 1, canvas, 1, 0)
    cv2.imshow("Smart Shape Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()