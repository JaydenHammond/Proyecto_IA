# importar librerias.
import cv2
import os

# Importar la clase
import SeguimientoManos as sm

# Creacion de la carpeta
nombre = 'Gesto_I'
direccion = 'C:/Users/Jayden/Downloads/ProyectoGestos/data'
carpeta = direccion + '/' + nombre

# si no esta creada la carpeta
if not os.path.exists(carpeta):
    print("Carpeta Creada: ", carpeta)
    os.makedirs(carpeta)

# lectura de la camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#declaramos contador
cont = 0

# declarar detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    # Realizar la lectura de la cam
    ret, frame = cap.read()

    #Extraer informacion de la mano
    frame = detector.encontrarmanos(frame, dibujar = False)

    #Posicion de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos= False, dibujarBox= False, color=[0,255,0])

    # si hat mano
    if mano == 1:
        #Extraer la informacion del cuadro
        xmin, ymin, xmax, ymax = bbox

        #Asignamos margen
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        # Limitar los valores al tamaño del frame
        alto, ancho, _ = frame.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(ancho, xmax)
        ymax = min(alto, ymax)

        # Validar que el recorte tenga sentido
        if xmax > xmin and ymax > ymin:
            recorte = frame[ymin:ymax, xmin:xmax]

            # Almacenar nuestra imagenes
            if recorte.size > 0:
                cv2.imwrite(carpeta + "/I_{}.jpg".format(cont), recorte)

                # aumentamos contador
                cont = cont + 1

                cv2.imshow('Recorte', recorte)

    # mostrar fps
    cv2.imshow('LENGUAJE VOCALES', frame)

    # leer nuestro teclado
    t = cv2.waitKey(1)
    if t == 27 or cont == 99:
        break

cap.release()
cv2.destroyAllWindows()
