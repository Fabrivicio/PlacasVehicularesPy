import cv2
import numpy as np
import pytesseract
from PIL import Image

captura = cv2.VideoCapture(0)  # Inicializa captura de video desde camara

while captura.isOpened():
    ret, frame = captura.read()

    if not ret:
        break

    al, an, c = frame.shape

    x1 = int(an / 3)
    x2 = int(x1 * 2)
    y1 = int(al / 3)
    y2 = int(y1 * 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    recorte = frame[y1:y2, x1:x2]

    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    Color = cv2.absdiff(mG, mB)

    umbral = cv2.threshold(Color, 40, 255, cv2.THRESH_BINARY)[1]  # Obtener el segundo valor retornado por cv2.threshold()
    umbral = np.array(umbral)  # Convertir a una matriz NumPy

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500 and area < 5000:
            x, y, ancho, alto = cv2.boundingRect(contorno)

            xpi = x + x1
            ypi = y + y1
            xpf = x + ancho + x1
            ypf = y + alto + y1

            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            placa = frame[ypi:ypf, xpi:xpf]

            alp, anp, cp = placa.shape
            Mva = np.zeros((alp, anp))

            mBp = np.matrix(placa[:, :, 0])
            mGp = np.matrix(placa[:, :, 1])
            mRp = np.matrix(placa[:, :, 2])

            for col in range(0, alp):
                for fil in range(0, anp):
                    Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                    Mva[col, fil] = 255 - Max

            _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")

            if alp >= 36 and anp >= 82:
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                config = "--psm 1"
                Texto = pytesseract.image_to_string(bin, config=config)

                if len(Texto) >= 7:
                    cv2.putText(frame, Texto, (xpi, ypi - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("vehiculos", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

captura.release()
cv2.destroyAllWindows()
