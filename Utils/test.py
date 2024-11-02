import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


# Squash Activation Function
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


# Capsule Layer Definition
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.W = self.add_weight(shape=[self.num_capsules, input_shape[-2], self.dim_capsule, self.input_dim],
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_hat = K.map_fn(lambda x: tf.linalg.matmul(self.W, x), elems=inputs_expand)
        inputs_hat_stopped = K.stop_gradient(inputs_hat)

        b = tf.zeros(shape=(K.shape(inputs)[0], self.num_capsules, inputs.shape[1]))
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.reduce_sum(c[:, :, tf.newaxis] * inputs_hat_stopped, axis=2))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat_stopped, [1, 3])

        return outputs


# Primary Capsule Layer
class PrimaryCap(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', dim_capsule=8, **kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu')
        self.dim_capsule = dim_capsule

    def call(self, inputs):
        # Convolutional Layer
        x = self.conv(inputs)
        # Reshape to primary capsules
        x = layers.Reshape((-1, self.dim_capsule))(x)
        return squash(x)


# Build the Capsule Network Model
def create_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(256, (9, 9), activation='relu')(input_layer)
    
    primary_caps = PrimaryCap(filters=256, kernel_size=(9, 9), strides=2, dim_capsule=8)(conv1)
    
    capsule = CapsuleLayer(num_capsules=10, dim_capsule=16, routings=3)(primary_caps)
    
    capsule_output = layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1)))(capsule)
    
    output_layer = layers.Dense(10, activation='softmax')(capsule_output)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# Main function to train and evaluate the model
def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create and compile the model
    model = create_model(input_shape=(28, 28, 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")


# Entry point
if __name__ == '__main__':
    main()
