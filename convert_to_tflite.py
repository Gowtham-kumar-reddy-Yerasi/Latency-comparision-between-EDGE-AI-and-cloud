import tensorflow as tf
import os
save_dir ='E://Sem3//Edge_AI_playgrounds//to git//Latency-comparision-between-EDGE-AI-and-cloud'
#load the model

model = tf.keras.models.load_model("mnist_model.keras")
print("loaded model mnist_model.keras")

#tflite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#quantization form FP32 TO INT8 4x size reduction

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = os.path.join(save_dir, 'model.tflite')

with open(tflite_path ,'wb') as f:
    f.write(tflite_model)


#compression report

sm_size = os.path.getsize('mnist_model.keras')/1024
tfl_size = os.path.getsize('model.tflite')/1024

print(f'\n Conversion complete!')
print(f'   SavedModel size  : {sm_size:,.1f} KB')
print(f'   TFLite size      : {tfl_size:,.1f} KB')
print(f'   Compression      : {sm_size/tfl_size:.1f}×')
