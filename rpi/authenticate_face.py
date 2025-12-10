import tensorflow as tf
import numpy as np
import os
import cv2

def prep_image(img):
    img = img.astype('float32')
    # Normalize to [-1, 1]
    img = (img - 127.5) / 128.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # shape: (1,112,112,3)
    return img

def get_embedding(img):
    interpreter = tf.lite.Interpreter(model_path="mobilefacenet_regular.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]['index'])

    return embedding

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b.T)

def authenticate_face(captured_image):
    
    captured_image = prep_image(captured_image)
    sample_embedding = get_embedding(captured_image)
    
    authorized_faces = []
    for image in os.listdir("authorized_faces"):
        img = cv2.imread(f"authorized_faces/{image}")
        if img is None:
            continue
        img = prep_image(img)
        print(img.shape)
        authorized_faces.append(get_embedding(img))
        
        
    for face in authorized_faces:
        similarity = cosine_similarity(sample_embedding, face)
        if (similarity > 0.6):
            return True, similarity
    
    return False, 0
