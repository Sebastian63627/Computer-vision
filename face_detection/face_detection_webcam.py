import cv2

# Cargar el clasificador Haarcascade incluido en OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la cámara (0 = cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error al abrir la cámara")
    exit()

print("✅ Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame")
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Dibujar rectángulos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con detecciones
    cv2.imshow("Detección de rostros", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
