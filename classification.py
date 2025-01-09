import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.squeeze()
y_test = y_test.squeeze()

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model



def create_unet_model():
    input_shape = (32, 32, 3)  # Adjust as needed
    num_classes = 10

    inputs = layers.Input(input_shape)

    # Encoder (downsampling)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder (upsampling)
    up1 = layers.UpSampling2D(size=(2, 2))(conv4)
    merge1 = layers.concatenate([up1, conv3])
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge1)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    merge2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge2)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up3 = layers.UpSampling2D(size=(2, 2))(conv6)
    merge3 = layers.concatenate([up3, conv1])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    # Global average pooling
    gap = layers.GlobalAveragePooling2D()(conv7)

# Output layer for classification
    outputs = layers.Dense(num_classes, activation='softmax')(gap)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
    

# Learning rate scheduler
def lr_schedule(epoch):
    lr = 0.0010
    if epoch > 75:
        lr *= 0.1
    elif epoch > 100:
        lr *= 0.01
    return lr

# Compile and train the models
def train_model(model, model_name):
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=1,  # Increased to 150 epochs
                        validation_split=0.1, batch_size=64, verbose=1,
                        callbacks=[lr_scheduler, early_stopping])
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")
    
    return history

# Run multiple training sessions
num_runs = 2
cnn_accuracies = []
unet_accuracies = []

for i in range(num_runs):
    print(f"Run {i+1}/{num_runs}")
    
    cnn_model = create_cnn_model()
    unet_model = create_unet_model()

    cnn_history = train_model(cnn_model, "CNN")
    unet_history = train_model(unet_model, "U-Net")
    
    cnn_accuracies.append(cnn_history.history['val_accuracy'][-1])
    unet_accuracies.append(unet_history.history['val_accuracy'][-1])

# Print average accuracies
print(f"Average CNN accuracy: {np.mean(cnn_accuracies):.4f} ± {np.std(cnn_accuracies):.4f}")
print(f"Average U-Net accuracy: {np.mean(unet_accuracies):.4f} ± {np.std(unet_accuracies):.4f}")

# Plot the training history for the last run
def plot_history(cnn_history, unet_history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['accuracy'], label='CNN Train')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation')
    plt.plot(unet_history.history['accuracy'], label='U-Net Train')
    plt.plot(unet_history.history['val_accuracy'], label='U-Net Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['loss'], label='CNN Train')
    plt.plot(cnn_history.history['val_loss'], label='CNN Validation')
    plt.plot(unet_history.history['loss'], label='U-Net Train')
    plt.plot(unet_history.history['val_loss'], label='U-Net Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(cnn_history, unet_history)


# Function to display original and predicted results and save as PNG
def display_predictions(model, x_test, y_test, model_name, num_samples=5):
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(x_test[i])
        plt.title(f"True: {y_test[i]}, Pred: {predicted_classes[i]}")
        plt.axis('off')
    
    plt.suptitle(f"{model_name} Predictions")
    plt.savefig(f"{model_name}_predictions.png")
    plt.show()

# Display and save predictions for both models
display_predictions(cnn_model, x_test, y_test, "CNN")
display_predictions(unet_model, x_test, y_test, "U-Net")

def get_feature_matrices(model, layer_name, input_image):
    feature_extractor = models.Model(inputs=model.inputs,
                                     outputs=model.get_layer(layer_name).output)
    features = feature_extractor.predict(input_image[np.newaxis, ...])
    return features[0]  # Return the feature matrices for the single input image

def plot_feature_matrices(features, layer_name, num_features=16, save_path=None):
    num_rows = int(np.ceil(np.sqrt(num_features)))
    num_cols = int(np.ceil(num_features / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    fig.suptitle(f'Feature Matrices for {layer_name}', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < min(num_features, features.shape[-1]):
            feature_matrix = features[:, :, i]
            sns.heatmap(feature_matrix, ax=ax, cmap='viridis', cbar=False)
            ax.set_title(f'Feature {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def print_feature_matrices(features, layer_name, num_features=2):
    for i in range(min(num_features, features.shape[-1])):
        feature_matrix = features[:, :, i]
        print(f"\nFeature Matrix {i+1} for {layer_name}:")
        
        # Convert the matrix to a string representation
        matrix_str = np.array2string(feature_matrix, precision=2, suppress_small=True, separator=' ', threshold=np.inf)
        print(matrix_str)

def get_conv_layers(model):
    return [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

def visualize_feature_matrices(model, model_name, input_image, output_type='show', output_dir=None):
    conv_layers = get_conv_layers(model)
    num_layers = len(conv_layers)
    layers_to_visualize = [conv_layers[0], conv_layers[num_layers // 2], conv_layers[-1]]
    
    for layer_name in layers_to_visualize:
        features = get_feature_matrices(model, layer_name, input_image)
        
        # Check if the feature matrix has only 2 dimensions, and add an extra dimension if necessary
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        
        if output_type == 'print':
            print_feature_matrices(features, f"{model_name} - {layer_name}")
        elif output_type == 'save':
            if output_dir is None:
                raise ValueError("output_dir must be specified when output_type is 'save'")
            save_path = f"{output_dir}/{model_name}_{layer_name}_feature_matrices.png"
            plot_feature_matrices(features, f"{model_name} - {layer_name}", save_path=save_path)
        else:  # 'show'
            plot_feature_matrices(features, f"{model_name} - {layer_name}")

sample_image = x_test[0]

# To print to command prompt
visualize_feature_matrices(cnn_model, "CNN", sample_image, output_type='print')
visualize_feature_matrices(unet_model, "U-Net", sample_image, output_type='print')

# To save as PNG
import os
output_dir = 'feature_matrices'
os.makedirs(output_dir, exist_ok=True)
visualize_feature_matrices(cnn_model, "CNN", sample_image, output_type='save', output_dir=output_dir)
visualize_feature_matrices(unet_model, "U-Net", sample_image, output_type='save', output_dir=output_dir)

# To show interactively (original behavior)
visualize_feature_matrices(cnn_model, "CNN", sample_image, output_type='show')
visualize_feature_matrices(unet_model, "U-Net", sample_image, output_type='show')