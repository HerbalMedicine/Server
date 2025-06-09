from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image

# Cargar el modelo ONNX
session = ort.InferenceSession('best.onnx')

# Nombres de las clases (copialo de model.names en Colab)
names = [
    'Alstonia_Scholaris_(P2)_diseased', 'Alstonia_Scholaris_(P2)_healthy', 'Arjun_(P1)_diseased', 'Arjun_(P1)_healthy', 
    'Chinar_(P11)_diseased', 'Chinar_(P11)_healthy', 'Gauva_(P3)_diseased', 'Gauva_(P3)_healthy', 'Jamun_(P5)_diseased', 
    'Jamun_(P5)_healthy', 'Jatropha_(P6)_diseased', 'Jatropha_(P6)_healthy', 'Lemon_(P10)_diseased', 'Lemon_(P10)_healthy', 
    'Mango_(P0)_diseased', 'Mango_(P0)_healthy', 'Pomegranate_(P9)_diseased', 'Pomegranate_(P9)_healthy', 
    'Pongamia_Pinnata_(P7)_diseased', 'Pongamia_Pinnata_(P7)_healthy'
]

# Obtener nombres de entrada/salida
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Leer imagen de la request
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    
    # Preprocesar la imagen como en el entrenamiento
    img = img.resize((224, 224))  # Tamaño que usaste en entrenamiento
    img = np.array(img).astype(np.float32) / 255.0  # Normalizar
    img = img.transpose(2, 0, 1)  # Convertir a (C, H, W)
    img = np.expand_dims(img, axis=0)  # (1, C, H, W)
    
    # Hacer predicción
    outputs = session.run([output_name], {input_name: img})[0]
    probs = outputs[0]  # (num_classes,)
    pred_index = np.argmax(probs)
    pred_clase = names[pred_index]
    confianza = probs[pred_index] * 100
    
    # Responder
    response = {
        'prediccion': pred_clase,
        'confianza': f'{confianza:.2f}%'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
