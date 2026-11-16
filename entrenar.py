import torch
from ultralytics import YOLO

if __name__ == '__main__':
    print("Usando dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Cargar el modelo ligero YOLOv8n
    model = YOLO('yolov8n.pt')

    # Entrenamiento optimizado para evitar congelamientos en Windows
    results = model.train(
        data='./dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,        # Reducido a 8 para liberar VRAM en la RTX 3050 Ti
        workers=0,      # OBLIGATORIO en Windows: evita saturar la memoria RAM
        name='pasos_peatonales_yolov8',
        device=0
    )

    # Exportar a TensorFlow Lite (.tflite) cuantizado
    model.export(format='tflite', int8=True)
    print("¡Entrenamiento y conversión completados exitosamente!")