import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

def extract_feature_matrices(model, layer_name, input_image):
    feature_extractor = tf.keras.models.Model(inputs=model.inputs,
                                              outputs=model.get_layer(layer_name).output)
    features = feature_extractor.predict(input_image[np.newaxis, ...])
    return features[0]

def normalize_feature_matrix(feature_matrix):
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature_matrix.reshape(-1, feature_matrix.shape[-1])).reshape(feature_matrix.shape)

def reduce_dimensionality(feature_matrix, n_components=50):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(feature_matrix.reshape(-1, feature_matrix.shape[-1]))

def compute_similarity_matrix(matrix1, matrix2):
    # Ensure both matrices have the same number of features
    min_features = min(matrix1.shape[1], matrix2.shape[1])
    matrix1 = matrix1[:, :min_features]
    matrix2 = matrix2[:, :min_features]
    return cosine_similarity(matrix1, matrix2)

def compute_correlation_matrix(matrix1, matrix2):
    # Ensure both matrices have the same number of features
    min_features = min(matrix1.shape[1], matrix2.shape[1])
    matrix1 = matrix1[:, :min_features]
    matrix2 = matrix2[:, :min_features]
    return np.corrcoef(matrix1.T, matrix2.T)

def visualize_similarity_matrix(similarity_matrix, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

def visualize_common_features(common_features, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(common_features, annot=False, cmap='Greens')
    plt.title(title)
    plt.show()

def compare_feature_matrices(cnn_model, unet_model, input_image, cnn_layer_name, unet_layer_name, similarity_threshold=0.9):
    # Extract feature matrices
    cnn_features = extract_feature_matrices(cnn_model, cnn_layer_name, input_image)
    unet_features = extract_feature_matrices(unet_model, unet_layer_name, input_image)
    
    # Normalize feature matrices
    cnn_features_norm = normalize_feature_matrix(cnn_features)
    unet_features_norm = normalize_feature_matrix(unet_features)
    
    # Reduce dimensionality to ensure same number of features
    n_components = min(cnn_features_norm.shape[-1], unet_features_norm.shape[-1])
    cnn_features_reduced = reduce_dimensionality(cnn_features_norm, n_components)
    unet_features_reduced = reduce_dimensionality(unet_features_norm, n_components)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(cnn_features_reduced, unet_features_reduced)
    
    # Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(cnn_features_reduced, unet_features_reduced)
    
    # Identify common features
    common_features = similarity_matrix > similarity_threshold
    
    # Visualize similarity matrix
    visualize_similarity_matrix(similarity_matrix, "Cosine Similarity Between CNN and U-Net Features")
    
    # Visualize correlation matrix
    visualize_similarity_matrix(correlation_matrix, "Correlation Between CNN and U-Net Features")
    
    # Visualize common features
    visualize_common_features(common_features, f"Common Features (Similarity > {similarity_threshold})")
    
    # Convert common features to DataFrame for better readability
    common_features_df = pd.DataFrame(common_features)
    print("Common features between CNN and U-Net:")
    print(common_features_df)
    
    return similarity_matrix, correlation_matrix, common_features_df

# Usage
sample_image = x_test[0]
cnn_layer_name = 'conv2d_5'  # Last convolutional layer in the CNN model
unet_layer_name = 'conv2d_11'  # Last convolutional layer in the U-Net model

similarity_matrix, correlation_matrix, common_features_df = compare_feature_matrices(
    cnn_model, unet_model, sample_image, cnn_layer_name, unet_layer_name, similarity_threshold=0.9
)

