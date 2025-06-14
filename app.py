
# caminho_img = (r"B:\Projetos VSCODE\Expouna_FaceID\pessoa.jpg")

# if not os.path.exists(caminho_img):
#     print(f"Imagem não encontrada em: {caminho_img}")
# else:
#     modelo = YOLO("yolov8n.pt")
#     results = modelo(r"B:\Projetos VSCODE\Expouna_FaceID\pessoa.jpg")
#     results[0].show()  


import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify

# Carrega o modelo YOLOv8 (nano)
modelo = YOLO("yolov8n.pt")

app = Flask(__name__)

# Variáveis globais
obj_detectado = False
camera_ligada = True

# Inicia a webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a webcam")
    exit()

# Função geradora para capturar frames da webcam
def gen_frames():
    global obj_detectado, camera_ligada

    while True:
        if not camera_ligada:
            # Frame preto com texto "Câmera desligada"
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Câmera desligada", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        ret, frame = cap.read()
        if not ret:
            break

        resultados = modelo(frame, verbose=False)[0]
        detectou = False

        for box in resultados.boxes:
            classe = int(box.cls[0])
            nome_classe = modelo.names[classe]

            # Verifica se a classe é uma das que queremos detectar
            if nome_classe == "person":
                detectou = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, nome_classe, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if nome_classe == "dog":
                detectou = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(frame, nome_classe, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

            if nome_classe == "car":
                detectou = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (153, 51, 153), 2)
                cv2.putText(frame, nome_classe, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 51, 153))

        obj_detectado = detectou
        print(f"Objeto detectado: {obj_detectado}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


##### ROTAS #####

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"obj_detectado": obj_detectado})

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_ligada
    camera_ligada = not camera_ligada
    return jsonify({"camera_ligada": camera_ligada})


if __name__ == '__main__':
    app.run(debug=True)



# cap.release()
# cv2.destroyAllWindows()

