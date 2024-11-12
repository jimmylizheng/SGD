from flask import Flask, request, jsonify, Response 
from io import BytesIO
import numpy as np
import struct
import json
import os
import time

app = Flask(__name__)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有跨域请求

scene_min = np.array([float('inf')] * 3)
scene_max = np.array([-float('inf')] * 3)

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
    

    for i in range(gaussian_count):
        offset = header_end + i * num_props * 4
        position = struct.unpack_from('<fff', content, offset)
        harmonic = struct.unpack_from('<fff', content, offset + 6 * 4)
        opacity_raw = struct.unpack_from('<f', content, offset + (6 + 48) * 4)[0]
        scale = struct.unpack_from('<fff', content, offset + (6 + 49) * 4)
        rotation = struct.unpack_from('<ffff', content, offset + (6 + 52) * 4)
        # # Normalize quaternion
        # rotation = np.array(rotation) / np.linalg.norm(rotation)

        # # Convert scale and rotation to covariance
        # scale = np.exp(scale)

        
        # 将 rotation 和 scale 转换为 float32 类型以匹配 JS 的 Float32Array
        rotation = np.array(rotation, dtype=np.float32)
        scale = np.array(scale, dtype=np.float32)
        # Normalize quaternion (手动实现与 JS 更接近)
        length2 = np.sum(rotation * rotation)
        length = np.sqrt(length2).astype(np.float32)  # 保证长度也是 float32
        rotation = rotation / length
        scale = np.exp(scale).astype(np.float32)
       



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

    def generate_batches(batch_size, gaussian_count):
        global scene_min, scene_max  # 声明这些变量是全局的
        # 数据解析，逐批次返回
        print(batch_size)
        print(gaussian_count)
        num_batches = (gaussian_count + batch_size - 1) // batch_size  # number of batches, round up
        print(f"Total batches: {num_batches}")

        for batch_index in range(num_batches):  # loop to generate batches
            start_index = batch_index * batch_size
            # ensure that the ending index will not exceed the total number of Gaussians
            end_index = min(start_index + batch_size, gaussian_count)

            # Get the splat attributes data needed
            opacities_batch = opacities[start_index:end_index].tolist()
            colors_batch = colors[3 * start_index: 3 * end_index].tolist()
            positions_batch = positions[3 * start_index: 3 * end_index]
            cov3ds_batch = cov3ds[6 * start_index: 6 * end_index]
            cov3ds_batch = [float(value) for value in cov3ds_batch]

            # get the scene min and scebe max of current batch
            # TODO: check the rendering pipeline of the usage of the scene min and scebe max
            for position in positions_batch:
                scene_min = np.minimum(scene_min, position)
                scene_max = np.maximum(scene_max, position)
            scene_min_batch = scene_min.tolist()
            scene_max_batch = scene_max.tolist()
            print('scenemin',scene_min)
            print(scene_min_batch)

            positions_batch = positions_batch.tolist()

            # 更新 count，表示已发送的点数
            count = end_index - start_index

            # 构造数据并通过 SSE 向客户端推送
            data = {
                'gaussians': {
                    'colors': colors_batch,
                    'cov3Ds': cov3ds_batch,
                    'opacities': opacities_batch,
                    'positions': positions_batch,
                    'count': count,  # number of splats in the current batch
                    'sceneMin': scene_min_batch,
                    'sceneMax': scene_max_batch
                }
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # 控制每次返回的时间间隔（根据需求调整）
            time.sleep(2)

    # 假设数据已经准备好，调用该函数生成批次
    return Response(generate_batches(gaussian_count // 5, gaussian_count), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
