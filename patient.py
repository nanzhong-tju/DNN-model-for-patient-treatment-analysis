import tensorflow as tf  
from tensorflow.keras import layers  

inputs = tf.keras.Input(shape=(10,))
x = layers.Dense(32, activation='relu', name='dense1')(inputs)
x = layers.Dense(64, activation='relu', name='dense2')(x)
x = layers.Dense(32, activation='relu', name='dense3')(x)
outputs = layers.Dense(2, name='beforesoftmax')(x)
outputs = layers.Softmax(name='predictions')(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

