from ultralytics import YOLO
import cv2
import torch

# Ruta del modelo que ya entrenaste
MODEL_PATH = r"D:\Servicio\Blind\Blind Code\runs\detect\pasos_peatonales_yolov8-4\weights\best.pt"

# Cargar modelo
model = YOLO(MODEL_PATH)

print(
    "Usando dispositivo:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

# 0 normalmente es la cámara principal.
# Si conectas una cámara USB/UVC y no aparece, después probaremos 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: No se pudo obtener imagen de la cámara.")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.50,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    frame_detectado = results[0].plot()

    cv2.imshow("Proyecto Blind - deteccion", frame_detectado)

    # Presiona Q para cerrar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()