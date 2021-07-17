import tensorflow as tf

###Load the model

model = tf.keras.models.load_model("rps_model.h5")

###Create the converter

converter = tf.lite.TFLiteConverter.from_keras_model(model)

###Convert the model to .tflite

tflite_model = converter.convert()
with open("rps.tflite", "wb") as f:
  f.write(tflite_model)