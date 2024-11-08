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
    print(gaussian_count)

    # positions = []
    # opacities = []
    # colors = []
    # cov3ds = []

    # Data parsing
    # for i in range(gaussian_count):
    #     offset = header_end + i * num_props * 4
    #     position = struct.unpack_from('<fff', content, offset)
    #     harmonic = struct.unpack_from('<fff', content, offset + 6 * 4)
    #     opacity_raw = struct.unpack_from('<f', content, offset + (6 + 48) * 4)[0]
    #     scale = struct.unpack_from('<fff', content, offset + (6 + 49) * 4)
    #     rotation = struct.unpack_from('<ffff', content, offset + (6 + 52) * 4)

    #     # Normalize quaternion
    #     rotation = np.array(rotation) / np.linalg.norm(rotation)

    #     # Convert scale and rotation to covariance
    #     cov3d = compute_cov3d(scale, 1, rotation)

    #     # Activate opacity
    #     opacity = sigmoid(opacity_raw)
        

    #     # Color based on harmonic
    #     sh_c0 = 0.28209479177387814
    #     color = [0.5 + sh_c0 * harmonic[0], 0.5 + sh_c0 * harmonic[1], 0.5 + sh_c0 * harmonic[2]]
    #     opacities = np.append(opacities, opacity)
    #     colors = np.append(colors, color)
    #     cov3ds = np.append(cov3ds, cov3d)
    #     positions = np.append(positions, position)
    # 预先分配数组
    opacities = np.zeros(gaussian_count)
    colors = np.zeros(3 * gaussian_count)  # 3倍长度
    cov3ds = []  # 6倍长度
    positions = np.zeros(3 * gaussian_count)  # 3倍长度
    scene_min = np.array([float('inf')] * 3)
    scene_max = np.array([-float('inf')] * 3)

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
        scale = np.exp(scale)

        if i == 0:
            print("First iteration - rotation:", rotation)
            print("First iteration - scale (before exp):", scale)   
        cov3d = compute_cov3d(scale, 1, rotation)

        # Activate opacity
        opacity = sigmoid(opacity_raw)

        # Color based on harmonic
        sh_c0 = 0.28209479177387814
        color = [0.5 + sh_c0 * harmonic[0], 0.5 + sh_c0 * harmonic[1], 0.5 + sh_c0 * harmonic[2]]

        # 填充数组
        opacities[i] = opacity

        # 将颜色和位置填充到一维数组中
        colors[3 * i: 3 * (i + 1)] = color
        positions[3 * i: 3 * (i + 1)] = position
        # 将 covariance 填充到一维数组中
        cov3ds.extend(cov3d)

        scene_min = np.minimum(scene_min, position)
        scene_max = np.maximum(scene_max, position)

    opacities_list = opacities.tolist()
    colors_list = colors.tolist()
    positions_list = positions.tolist()
    scene_min_list = scene_min.tolist()
    scene_max_list = scene_max.tolist()

    # Package processed data
    print("successfully processed data")
    return jsonify({
        'gaussians': {
            'colors': colors_list,
            'cov3Ds': cov3ds,
            'opacities':opacities_list,
            'positions':positions_list,
            'count': gaussian_count,
            'sceneMin': scene_min_list,
            'sceneMax': scene_max_list
        },  # 这里是四个数组
    })

if __name__ == '__main__':
    app.run(debug=True)
