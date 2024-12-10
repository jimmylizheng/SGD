import os
import struct
import json
import numpy as np


scene_min = np.array([float('inf')] * 3)
scene_max = np.array([-float('inf')] * 3)

def sort_by_scale(opacities, colors, positions, cov3ds, scales):
    """
    sort based on the splat's size
    
    params:
        opacities (np.ndarray): length of n
        colors (np.ndarray): length of 3 * n
        positions (np.ndarray): length of 3 * n
        cov3ds (list): length of 6 * n
        scales (np.ndarray): length of 3 * n (every 3 values is a splat's [x, y, z])
        
    return:
        tuple: sorted (opacities, colors, positions, cov3ds, scales)
    """
    # calculate the utility of each splat U = x * y * z
    scales_reshaped = scales.reshape(-1, 3)  # reshape the scales to [n, 3]
    utility_values = np.prod(scales_reshaped, axis=1)  # calculate the size of each splat
    
    # get the sorted index
    sorted_indices = np.argsort(utility_values)[::-1]
    
    # sort the data based on the index
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors.reshape(-1, 3)[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    sorted_scales = scales_reshaped[sorted_indices].flatten()
    
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds, sorted_scales

def sort_by_brightness(opacities, colors, positions, cov3ds):
    """
    sort based on the splat brightness
    
    params:
        opacities (np.ndarray): length of n
        colors (np.ndarray): length of 3 * n
        positions (np.ndarray): length of 3 * n
        cov3ds (list): length of 6 * n
        
    return:
        tuple: sorted (opacities, colors, positions, cov3ds)
    """
    # get the RGB values
    colors_reshaped = colors.reshape(-1, 3)  # reshape the colors to [n, 3]
    
    # calculate the brightness
    brightness = (
        0.299 * colors_reshaped[:, 0] +  # R
        0.587 * colors_reshaped[:, 1] +  # G
        0.114 * colors_reshaped[:, 2]    # B
    )
    
    # get the sorted index
    sorted_indices = np.argsort(brightness)[::-1]
    
    # sort the data based on the index
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors_reshaped[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds

def sort_data(opacities, colors, positions, cov3ds):
    """
    sorting based on the opacities (descending)
    
    params:
        opacities (np.ndarray): length of n
        colors (np.ndarray): length of 3 * n
        positions (np.ndarray): length of 3 * n
        cov3ds (list): length of 6 * n
        
    return:
        tuple: sorted (opacities, colors, positions, cov3ds)
    """
    # the the sorting index
    sorted_indices = np.argsort(opacities)[::-1]
    
    # resort the data based on the utility function
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors.reshape(-1, 3)[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cov3d(scale, mod, rot):
    # Compute scaling matrix
    S = np.diag([mod * scale[0], mod * scale[1], mod * scale[2]])

    # Quaternion to rotation matrix
    r, x, y, z = rot
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
        [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
        [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)]
    ])

    # Compute 3D world covariance matrix Sigma
    # M = S @ R
    M = R @ S
    # Sigma = M.T @ M
    Sigma = M @ M.T

    # Covariance is symmetric, only store upper triangular part
    cov3d = [Sigma[0, 0], Sigma[0, 1], Sigma[0, 2], Sigma[1, 1], Sigma[1, 2], Sigma[2, 2]]
    return cov3d

def process_ply_file(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")

    print("Reading ply file")
    with open(input_file, 'rb') as file:
        content = file.read()

    # Parse header and Gaussian data
    header_end = content.find(b'end_header') + len(b'end_header') + 1
    header = content[:header_end].decode('utf-8')
    gaussian_count = int(next(line.split()[-1] for line in header.splitlines() if line.startswith('element vertex')))
    num_props = 62  # Total properties per Gaussian
    #gaussian_count = gaussian_count // 50  # Debugging, scale down for experiments

    opacities = np.zeros(gaussian_count)
    colors = np.zeros(3 * gaussian_count)
    cov3ds = []
    positions = np.zeros(3 * gaussian_count)
    scales = np.zeros(3 * gaussian_count)

    for i in range(gaussian_count):
        offset = header_end + i * num_props * 4
        position = struct.unpack_from('<fff', content, offset)
        harmonic = struct.unpack_from('<fff', content, offset + 6 * 4)
        opacity_raw = struct.unpack_from('<f', content, offset + (6 + 48) * 4)[0]
        scale = struct.unpack_from('<fff', content, offset + (6 + 49) * 4)
        rotation = struct.unpack_from('<ffff', content, offset + (6 + 52) * 4)

        rotation = np.array(rotation, dtype=np.float32)
        scale = np.array(scale, dtype=np.float32)
        length = np.sqrt(np.sum(rotation * rotation)).astype(np.float32)
        rotation /= length
        scale = np.exp(scale).astype(np.float32)
        cov3d = compute_cov3d(scale, 1, rotation)
        opacity = sigmoid(opacity_raw)

        sh_c0 = 0.28209479177387814
        color = [0.5 + sh_c0 * harmonic[0], 0.5 + sh_c0 * harmonic[1], 0.5 + sh_c0 * harmonic[2]]
        opacities[i] = opacity
        colors[3 * i: 3 * (i + 1)] = color
        positions[3 * i: 3 * (i + 1)] = position
        scales[3 * i: 3 * (i + 1)] = scale
        cov3ds.extend(cov3d)

    print("Finish preprocessing")

    # opacities, colors, positions, cov3ds = sort_by_brightness(opacities, colors, positions, cov3ds) # opacity

    opacities, colors, positions, cov3ds, scales = sort_by_scale(opacities, colors, positions, cov3ds, scales) # opacityale
    
    # Convert all numpy arrays to native Python types for JSON serialization
    data = {
        "opacities": opacities.astype(float).tolist(),
        "colors": colors.astype(float).tolist(),
        "positions": positions.astype(float).tolist(),
        "cov3ds": [float(value) for value in cov3ds],
        "gaussian_count": int(gaussian_count)  # Ensure integer type
    }

    # Save data to JSON
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_ply_file = "room.ply"  # Replace with your .ply file
    # output_json_file = "rooms.json"  
    # output_json_file = "opacity_rooms.json"  
    output_json_file = "splats_rooms.json"
    # output_json_file = "brightness_rooms.json"

    try:
        process_ply_file(input_ply_file, output_json_file)
    except FileNotFoundError as e:
        print(e)