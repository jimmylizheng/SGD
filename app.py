from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
import struct
import json
import os

app = Flask(__name__)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有跨域请求


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
    M = S @ R
    Sigma = M.T @ M

    # Covariance is symmetric, only store upper triangular part
    cov3d = [Sigma[0, 0], Sigma[0, 1], Sigma[0, 2], Sigma[1, 1], Sigma[1, 2], Sigma[2, 2]]
    return cov3d

@app.route('/api/load_scene', methods=['GET'])
def load_scene():
    
    # 获取查询参数中的 scene 名称
    scene = request.args.get('scene')
    print("Msg Recd", scene)
    if not scene:
        return jsonify({"error": "Scene parameter is required"}), 400

    # 构建文件路径
    filename = f"{scene}.ply"
    if not os.path.exists(filename):
        return jsonify({"error": f"File '{filename}' not found"}), 404
    
    print("Reading ply file")

    # Read the scene file
    with open(filename, 'rb') as file:
        content = file.read()

    # Find header and data section
    header_end = content.find(b'end_header') + len(b'end_header') + 1
    header = content[:header_end].decode('utf-8')

    # Extract gaussian count
    gaussian_count = int(next(line.split()[-1] for line in header.splitlines() if line.startswith('element vertex')))
    num_props = 62  # Total properties per Gaussian

    positions = []
    opacities = []
    colors = []
    cov3ds = []

    # Data parsing
    for i in range(gaussian_count):
        offset = header_end + i * num_props * 4
        position = struct.unpack_from('<fff', content, offset)
        harmonic = struct.unpack_from('<fff', content, offset + 6 * 4)
        opacity_raw = struct.unpack_from('<f', content, offset + (6 + 48) * 4)[0]
        scale = struct.unpack_from('<fff', content, offset + (6 + 49) * 4)
        rotation = struct.unpack_from('<ffff', content, offset + (6 + 52) * 4)

        # Normalize quaternion
        rotation = np.array(rotation) / np.linalg.norm(rotation)

        # Convert scale and rotation to covariance
        cov3d = compute_cov3d(scale, 1, rotation)

        # Activate opacity
        opacity = sigmoid(opacity_raw)
        opacities.append(opacity)

        # Color based on harmonic
        sh_c0 = 0.28209479177387814
        color = [0.5 + sh_c0 * harmonic[0], 0.5 + sh_c0 * harmonic[1], 0.5 + sh_c0 * harmonic[2]]
        colors.append(color)
        cov3ds.append(cov3d)
        positions.append(position)

    # Package processed data
    print("successfully processed data")
    return jsonify({
        'gaussians': [positions, opacities, colors, cov3ds],  # 这里是四个数组
        'count': gaussian_count
    })

if __name__ == '__main__':
    app.run(debug=True)
