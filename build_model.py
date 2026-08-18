import tensorflow as tf
import numpy as np
import os

current_work = "E://Sem3//Edge_AI_playgrounds//to git//Latency-comparision-between-EDGE-AI-and-cloud"

(x_train,y_train),(x_test,y_test) =tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0

print(f'Training samples : {len(x_train):,}')
print(f'Test samples : {len(x_test):,}')

#Model_Architecture

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8,kernel_size = 3,activation= 'relu',padding='same',input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation ='relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
],name = 'mnist_cnn')


model.summary()

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(x_train,y_train,epochs =5 ,batch_size = 64,validation_split=0.1,verbose =1)

loss,acc = model.evaluate(x_test,y_test,verbose = 0)
print(f'\nTest accuracy : {acc:.4f}  ({acc*100:.2f}%)')


model_path = os.path.join(current_work, 'mnist_model.keras')
sample_path = os.path.join(current_work, 'test_sample.npy')
# ── 6. Save ────────────────────────────────────────────────
model.save(model_path)
print('\n Model saved to  mnist_model/')
 
# Save one test sample for consistent benchmarking later
np.save(sample_path, x_test[:1])
print(f' Test sample saved  (true label: {y_test[0]})')
