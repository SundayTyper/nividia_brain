import matplotlib.pyplot as plt
import tensorflow as tf 

tf.config.list_physical_devices('GPU')

# define data variables for training. These are included in tensorflow. Images of fashion items
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

# number of neurons
number_of_classes = train_labels.max() + 1

# model type for making predictions. Flatten to read, Dense to calculate
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(number_of_classes)
])

# define what I want to see as the trainer
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training stage
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
    )

# work on a batch of data
model.predict(train_images[0:10])
