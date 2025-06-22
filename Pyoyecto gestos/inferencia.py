import cv2
import os
import subprocess
import time
from ultralytics import YOLO
import SeguimientoManos as sm

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Cargar modelo
model = YOLO('best (1).pt')

# Detector de manos
detector = sm.detectormanos(Confdeteccion=0.5)

# Control de gesto
ultimo_gesto = None
tiempo_inicio = 0
gesto_ejecutado = None

while True:
    ret, frame = cap.read()
    frame = detector.encontrarmanos(frame, dibujar=False)
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False)

    if mano == 1:
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin = max(0, xmin - 40), max(0, ymin - 40)
        xmax, ymax = min(frame.shape[1], xmax + 40), min(frame.shape[0], ymax + 40)

        if xmax > xmin and ymax > ymin:
            recorte = frame[ymin:ymax, xmin:xmax]

            # Confianza aumentada para evitar falsos positivos
            resultados = model.predict(recorte, conf=0.75)

            if resultados and len(resultados[0].boxes) > 0:
                r = resultados[0]
                class_id = int(r.boxes.cls[0].item())
                nombre_clase = r.names[class_id]

                # Mostrar texto SOLO si no es Gesto_Rock
                if nombre_clase != 'Gesto_Rock':
                    cv2.putText(frame, f"Gesto: {nombre_clase}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Iniciar conteo de tiempo si es un nuevo gesto
                if nombre_clase != ultimo_gesto:
                    ultimo_gesto = nombre_clase
                    tiempo_inicio = time.time()
                else:
                    tiempo_gesto = time.time() - tiempo_inicio
                    if tiempo_gesto >= 2 and gesto_ejecutado != nombre_clase and nombre_clase != 'Gesto_Rock':
                        print(f"🖐 Ejecutando acción para: {nombre_clase}")
                        gesto_ejecutado = nombre_clase

                        if nombre_clase == 'Gesto_I':
                            subprocess.Popen(['notepad.exe'])

                        elif nombre_clase == 'Gesto_puno':
                            subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'])

                        elif nombre_clase == 'Gesto perfecto':
                            subprocess.Popen(['calc.exe'])

                        elif nombre_clase == 'Gesto_Pistola':
                            subprocess.Popen(['mspaint.exe'])

    else:
        # Reiniciar estados si no hay mano
        ultimo_gesto = None
        gesto_ejecutado = None
        tiempo_inicio = 0

    cv2.imshow('LENGUAJE GESTUAL', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
