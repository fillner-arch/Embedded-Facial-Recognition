import tensorflow as tf
import numpy as np
from PIL import Image
import time

def prepare_image(img_path):
    # Load and resize image to 112x112
    img = Image.open(img_path).convert("RGB").resize((112,112))
    img = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1]
    img = (img - 127.5) / 128.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # shape: (1,112,112,3)
    return img

def get_embedding(interpreter, img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]['index'])

    return embedding

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


# start = time.time()

interpreter = tf.lite.Interpreter(model_path="mobilefacenet_regular.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


img1 = prepare_image("./images/1.jpeg")
img2 = prepare_image("./images/2.jpeg")
img3 = prepare_image("./images/3.png")

embedding1 = get_embedding(interpreter, img1)
embedding2 = get_embedding(interpreter, img2)
embedding3 = get_embedding(interpreter, img3)
# cpu_frequency = psutil.cpu_freq()

# Example: compare two faces
# sim = cosine_similarity(embedding1[0], embedding2[0])
# sim = cosine_similarity(embedding1[0], embedding3[0])
# sim = cosine_similarity(embedding2[0], embedding3[0])

# end = time.time()

# print(f"time to complete: {end-start}")
# print(f"Current CPU frequency: {cpu_frequency.current} MHz")
# print(f"Minimum CPU frequency: {cpu_frequency.min} MHz")
# print(f"Maximum CPU frequency: {cpu_frequency.max} MHz")

