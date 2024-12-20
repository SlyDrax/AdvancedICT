import argparse
import laspy
import numpy as np
import os

def augment_block(block):
    augmented_blocks = []
    for angle in [0, 90, 180, 270]:
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        rotated_block = block.copy()
        rotated_block[:, :3] = np.dot(rotated_block[:, :3], rotation_matrix.T)
        augmented_blocks.append(rotated_block)

        # Add mirrored versions (flip along X and Y axes)
        # mirrored_x = rotated_block.copy()
        # mirrored_x[:, 0] *= -1  # Flip X
        # augmented_blocks.append(mirrored_x)

        # mirrored_y = rotated_block.copy()
        # mirrored_y[:, 1] *= -1  # Flip Y
        # augmented_blocks.append(mirrored_y)

    return augmented_blocks

def create_blocks(points, labels, block_size=50.0, stride=25.0):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    blocks = []
    block_labels = []
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            x_cond = (points[:, 0] >= x) & (points[:, 0] < x + block_size)
            y_cond = (points[:, 1] >= y) & (points[:, 1] < y + block_size)
            cond = x_cond & y_cond

            sub_points = points[cond]
            sub_labels = labels[cond]

            if sub_points.shape[0] > 0:
                centroid = np.mean(sub_points[:, :3], axis=0)  # Center
                sub_points[:, :3] -= centroid
                dist = np.max(np.linalg.norm(sub_points[:, :3], axis=1))
                if dist > 0:
                    sub_points[:, :3] /= dist

                augmented_blocks = augment_block(sub_points)
                blocks.extend(augmented_blocks)
                block_labels.extend([sub_labels] * len(augmented_blocks))
                # blocks.append(sub_points)
                # block_labels.append(sub_labels)

            y += stride
        x += stride

    return blocks, block_labels

def process_las_file(input_path, output_folder, block_size, stride, use_intensity):
    class_map = {
        1: 0,  # Terrain
        2: 0,  # Terrain
        3: 0,  # Terrain
        4: 0,  # Terrain
        5: 0,  # Terrain
        6: 0,  # Terrain
        7: 0,  # grimstad building
        8: 1,  # dales building
    }

    print(f"Reading LAS file: {input_path}")
    las = laspy.read(input_path)
    x, y, z = las.x, las.y, las.z

    las_labels = getattr(las, 'classification', np.zeros_like(x, dtype=np.uint8))
    mapped_labels = np.array([class_map.get(lbl, 0) for lbl in las_labels], dtype=np.int64)

    if use_intensity and hasattr(las, 'intensity'):
        intensity = las.intensity
        points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
    else:
        points = np.stack([x, y, z], axis=-1).astype(np.float32)

    blocks, block_labels = create_blocks(points, mapped_labels, block_size=block_size, stride=stride)

    block_arrays = np.array(blocks, dtype=object)
    label_arrays = np.array(block_labels, dtype=object)

    output_filename = os.path.join(
        output_folder,
        f"{os.path.splitext(os.path.basename(input_path))[0]}_block_size={int(block_size)}_stride={int(stride)}.npz"
    )
    np.savez(output_filename, block_points=block_arrays, block_labels=label_arrays)
    print(f"Saved preprocessed data to {output_filename}")
    print(f"Number of blocks: {len(blocks)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="data/labeled/", help="Folder containing input .las files.")
    parser.add_argument("--output_folder", type=str, default="preprocessed_test", help="Folder to save preprocessed .npz files.")
    parser.add_argument("--block_size", type=float, default=50.0, help="Block size in XY plane.")
    parser.add_argument("--stride", type=float, default=20.0, help="Stride for overlapping blocks.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    las_files = [f for f in os.listdir(args.input_folder) if f.endswith('.las')]
    print(f"Found {len(las_files)} LAS files in folder: {args.input_folder}")

    for las_file in las_files:
        input_path = os.path.join(args.input_folder, las_file)
        process_las_file(input_path, args.output_folder, args.block_size, args.stride, False)

    print("Processing complete.")

if __name__ == "__main__":
    main()
