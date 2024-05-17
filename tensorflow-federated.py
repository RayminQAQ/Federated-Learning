import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import threading
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import tensorflow_federated as tff

# Set the number of clients participating in the federated learning.
client_number = 3

def read_images_from_png(file_path):
    """Reads and decodes images from a given file path."""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    return image.numpy()

def build_model(num_classes):
    """Builds a TensorFlow Keras model suitable for federated learning."""
    model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))
    ])
    return model

def get_keras_model_from_tff_weights(model_fn, server_state):
    # Create a new instance of the Keras model
    # Ensure num_classes matches the number used during federated training
    keras_model = build_model(num_classes=12)  # Updated to match the trained model's classes

    # Extract the trainable and non-trainable weights from the TFF model state
    tff_weights = server_state.model.trainable + server_state.model.non_trainable

    # Assign these weights to the Keras model
    keras_model.set_weights(tff_weights)

    return keras_model
def client_optimizer_fn():
    """Returns a differentially private optimizer configured with specific privacy parameters."""
    # The clipping norm and noise multiplier are key parameters for differential privacy.
    l2_norm_clip = 1.0
    noise_multiplier = 1.1
    num_microbatches = 1  # Can be set to the batch size if each sample is a microbatch
    learning_rate= 0.02
    
    # Create a differentially private SGD optimizer using TensorFlow Privacy.
    dp_optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate= learning_rate
    )
    return dp_optimizer
class MyClientData:
    def __init__(self, client_data):
        self.client_data = client_data

    def create_tf_dataset_for_client(self, client_id):
        return self.client_data[client_id]

    @property
    def client_ids(self):
        return list(self.client_data.keys())

def model_fn():
    """Defines how to construct the model and its optimizer in the federated setting."""
    return tff.learning.from_keras_model(
        keras_model=build_model(num_classes=12),
        input_spec=federated_data.create_tf_dataset_for_client(federated_data.client_ids[0]).element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

def load_data(folder):
    client_data = {}
    label_mapping = {}  # Dictionary to map categories to integers
    next_label = 0  # Counter to assign new labels

    # Iterate over category directories inside the base folder
    for category in os.listdir(folder):
        category_dir = os.path.join(folder, category)
        if not os.path.isdir(category_dir):
            continue  # Skip if not a directory

        # Assign a numeric label to each unique category
        if category not in label_mapping:
            label_mapping[category] = next_label
            next_label += 1
            
        # Collect all images in this category
        all_data = []
        all_labels = []
        for filename in os.listdir(category_dir):
            if filename.endswith(".png"):
                file_path = os.path.join(category_dir, filename)
                image = read_images_from_png(file_path)
                all_data.append(image)
                all_labels.append(label_mapping[category])
                
        # If there are images, distribute them across clients
        if all_data:
            # Convert lists to numpy arrays and normalize pixel values
            all_data = np.array(all_data, dtype=np.uint8).reshape(-1, 28, 28, 1) / 255.0
            all_labels = np.array(all_labels)

            # Shuffle the data and labels in the same order
            indices = np.random.permutation(len(all_data))
            all_data = all_data[indices]
            all_labels = all_labels[indices]

            # Divide the data into chunks for each client
            data_chunks = np.array_split(all_data, client_number)
            label_chunks = np.array_split(all_labels, client_number)

            for client_id in range(client_number):
                if len(data_chunks[client_id]) > 0 and len(label_chunks[client_id]) > 0:
                    dataset = tf.data.Dataset.from_tensor_slices(
                        (data_chunks[client_id], label_chunks[client_id])
                        )
                    dataset = dataset.shuffle(len(data_chunks[client_id]))
                    dataset = dataset.batch(16)
                    client_data[f'client_{client_id + 1}'] = dataset
                else:
                    print(f"No data available for client {client_id + 1}")
    return client_data


def run_federated_learning():
    # Set the local execution context for TensorFlow Federated
    #tff.backends.native.set_local_execution_context()

    # Load data and initialize federated data
    global federated_data
    base_folder = "image_fromTA"
    client_data = load_data(base_folder)
    federated_data = MyClientData(client_data)

    # Build the federated averaging process
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.02, epsilon=10),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0, epsilon=10)
    )

    # Initialize and run the federated learning process
    state = iterative_process.initialize()
    num_rounds = 100
    num_clients_per_round = client_number
    
    metrics_history = []
    for round_num in range(1, num_rounds + 1):
        selected_clients = np.random.choice(federated_data.client_ids, size=num_clients_per_round, replace=False)
        federated_train_data = [federated_data.create_tf_dataset_for_client(x) for x in selected_clients]
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f'Round {round_num:2d}, Metrics: {metrics}')
        metrics_history.append(metrics)

    accuracies = [metrics['train']['sparse_categorical_accuracy'] for metrics in metrics_history]
    final_accuracy = np.mean(accuracies)
    print(f"Final averaged accuracy over {len(metrics_history)} rounds is: {final_accuracy}")
    
    final_model = get_keras_model_from_tff_weights(model_fn, state)
    # save as TensorFlow SavedModel
    final_model.save('model'+str(num_rounds)+'.h5')
    
run_federated_learning()

# epsilon = 100 -> Accuracy: 0.11784865707159042
# epsilon = 50 -> Accuracy: 0.16763243079185486
# epsilon = 10 -> Accuracy: 0.09182702749967575
# epsilon = 1e-06 -> Accuracy: 0.9792972803115845
# epsilon = 1e-08 -> Accuracy: 0.9804324507713318
# epsilon = 0 -> Accuracy: 0.9815567135810852