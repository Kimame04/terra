import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

def get_category(img):
    img = mpimg.imread(img)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = np.expand_dims(img, axis=0)[:, :, :, :3]

    tflite_model_file = 'static/model/model.tflite'
    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    prediction = []
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    prediction.append(interpreter.get_tensor(output_index))

    predicted_label = np.argmax(prediction)
    class_names = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt')
    return dict(zip(class_names, interpreter.get_tensor(output_index)[0].tolist())), list(class_names)[predicted_label]