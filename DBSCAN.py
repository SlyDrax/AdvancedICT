import os
import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

input_directory = "dataset"
classified_directory = "classified"

os.makedirs(classified_directory, exist_ok=True)

def compute_features(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    height_normalized = z - np.min(z)

    tree = KDTree(points[:, :3])
    density = np.array([len(tree.query_ball_point(pt, r=1.0)) for pt in points[:, :3]])

    curvatures = []
    epsilon = 1e-6  
    for pt in points[:, :3]:
        neighbors = tree.query_ball_point(pt, r=1.0)
        if len(neighbors) > 1:
            covariance = np.cov(points[neighbors, :3] - pt, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance)
            curvature = eigenvalues.min() / (eigenvalues.sum() + epsilon)
            curvatures.append(curvature)
        else:
            curvatures.append(0)

    curvatures = np.array(curvatures)

    features = np.column_stack((x, y, z, height_normalized, density, curvatures))

    features = features[~np.isnan(features).any(axis=1)]

    return features

def apply_dbscan(features):
    dbscan = DBSCAN(eps=2.0, min_samples=10)
    labels = dbscan.fit_predict(features)
    return labels

def remap_labels(labels):
    remapped_labels = np.zeros_like(labels, dtype=np.uint32)
    noise_value = 255

    max_label = labels.max()
    for original_label in range(-1, max_label + 1):
        if original_label == -1:  # Noise
            remapped_labels[labels == original_label] = noise_value
        else:
            remapped_labels[labels == original_label] = original_label + (original_label // 255)
    return remapped_labels

for file_name in os.listdir(input_directory):
    if file_name.endswith(".laz"):
        print(f"Processing file: {file_name}")
        input_path = os.path.join(input_directory, file_name)

        las = laspy.read(input_path)
        points = np.vstack((las.x, las.y, las.z)).T

        features = compute_features(points)

        if features.size == 0:
            print(f"Skipping file {file_name}: No valid features after NaN removal.")
            continue

        labels = apply_dbscan(features)

        labels = remap_labels(labels)

        classified_file_path = os.path.join(classified_directory, file_name.replace(".laz", "_classified.las"))
        las.classification = labels.astype(np.uint8)
        las.write(classified_file_path)

        print(f"Classified file saved to {classified_file_path}")

print("Batch processing and classification completed.")
