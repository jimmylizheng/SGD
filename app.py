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
CORS(app)  # allow cross domain request?

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
    # M = S @ R
    M = R @ S
    # Sigma = M.T @ M
    Sigma = M @ M.T

    # Covariance is symmetric, only store upper triangular part
    cov3d = [Sigma[0, 0], Sigma[0, 1], Sigma[0, 2], Sigma[1, 1], Sigma[1, 2], Sigma[2, 2]]
    return cov3d

def sort_by_opacity_and_size(opacities, colors, positions, cov3ds, scales):
    """
    根据透明度和大小的组合效用值降序排序，并同步调整其他数据。
    
    参数:
        opacities (np.ndarray): 不透明度数组，长度为 n。
        colors (np.ndarray): 颜色数组，长度为 3 * n。
        positions (np.ndarray): 位置数组，长度为 3 * n。
        cov3ds (list): 协方差数组，长度为 6 * n。
        scales (np.ndarray): 尺度数组，长度为 3 * n。
        
    返回:
        tuple: 排序后的 (opacities, colors, positions, cov3ds, scales)。
    """
    # 计算每个 splat 的大小：x * y * z
    scales_reshaped = scales.reshape(-1, 3)  # 将 scales 变为 [n, 3] 的形状
    size = np.prod(scales_reshaped, axis=1)  # 计算每个 splat 的大小
    
    # 计算效用值 U = opacity * size
    utility_values = opacities * size
    
    # 获取按效用值降序的索引
    sorted_indices = np.argsort(utility_values)[::-1]
    
    # 按索引重排数据
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors.reshape(-1, 3)[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    sorted_scales = scales_reshaped[sorted_indices].flatten()
    
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds, sorted_scales


def sort_by_brightness(opacities, colors, positions, cov3ds):
    """
    根据 splat 的亮度值降序排序，并同步调整其他数据。
    
    参数:
        opacities (np.ndarray): 不透明度数组，长度为 n。
        colors (np.ndarray): 颜色数组，长度为 3 * n。
        positions (np.ndarray): 位置数组，长度为 3 * n。
        cov3ds (list): 协方差数组，长度为 6 * n。
        
    返回:
        tuple: 排序后的 (opacities, colors, positions, cov3ds)。
    """
    # 提取 RGB 值
    colors_reshaped = colors.reshape(-1, 3)  # 将 colors 变为 [n, 3] 的形状
    
    # 根据公式计算亮度值
    brightness = (
        0.299 * colors_reshaped[:, 0] +  # R 分量
        0.587 * colors_reshaped[:, 1] +  # G 分量
        0.114 * colors_reshaped[:, 2]    # B 分量
    )
    
    # 获取按亮度值降序的索引
    sorted_indices = np.argsort(brightness)[::-1]
    
    # 按索引重排数据
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors_reshaped[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds


def sort_by_scale(opacities, colors, positions, cov3ds, scales):
    """
    根据 splat 的大小 (scale 的乘积) 降序排序，并同步调整其他数据。
    
    参数:
        opacities (np.ndarray): 不透明度数组，长度为 n。
        colors (np.ndarray): 颜色数组，长度为 3 * n。
        positions (np.ndarray): 位置数组，长度为 3 * n。
        cov3ds (list): 协方差数组，长度为 6 * n。
        scales (np.ndarray): 尺度数组，长度为 3 * n (每 3 个值为一个 splat 的 [x, y, z])。
        
    返回:
        tuple: 排序后的 (opacities, colors, positions, cov3ds, scales)。
    """
    # 计算每个 splat 的效用值 U = x * y * z
    scales_reshaped = scales.reshape(-1, 3)  # 将 scales 变为 [n, 3] 的形状
    utility_values = np.prod(scales_reshaped, axis=1)  # 计算每个 splat 的大小
    
    # 获取按效用值降序的索引
    sorted_indices = np.argsort(utility_values)[::-1]
    
    # 按索引重排数据
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors.reshape(-1, 3)[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    sorted_scales = scales_reshaped[sorted_indices].flatten()
    
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds, sorted_scales


def sort_data(opacities, colors, positions, cov3ds):
    """
    根据 opacities 降序对数据进行排序，并同步调整 colors、positions 和 cov3ds。
    
    参数:
        opacities (np.ndarray): 不透明度数组，长度为 n。
        colors (np.ndarray): 颜色数组，长度为 3 * n。
        positions (np.ndarray): 位置数组，长度为 3 * n。
        cov3ds (list): 协方差数组，长度为 6 * n。
        
    返回:
        tuple: 排序后的 (opacities, colors, positions, cov3ds)。
    """
    # 获取降序排序索引
    sorted_indices = np.argsort(opacities)[::-1]
    
    # 根据排序索引重排数据
    sorted_opacities = opacities[sorted_indices]
    sorted_colors = colors.reshape(-1, 3)[sorted_indices].flatten()
    sorted_positions = positions.reshape(-1, 3)[sorted_indices].flatten()
    sorted_cov3ds = np.array(cov3ds).reshape(-1, 6)[sorted_indices].flatten().tolist()
    return sorted_opacities, sorted_colors, sorted_positions, sorted_cov3ds


# @app.route('/api/load_scene', methods=['GET'])
# def load_scene():
    
#     # get the name of the scene from the args of the request
#     scene = request.args.get('scene')
#     print("Msg Recd", scene)
#     if not scene:
#         return jsonify({"error": "Scene parameter is required"}), 400

#     # construct the file name
#     filename = f"{scene}.ply"
#     if not os.path.exists(filename):
#         return jsonify({"error": f"File '{filename}' not found"}), 404
    
#     print("Reading ply file")

#     # Read the scene file
#     with open(filename, 'rb') as file:
#         content = file.read()

#     # Find header and data section
#     header_end = content.find(b'end_header') + len(b'end_header') + 1
#     header = content[:header_end].decode('utf-8')

#     # Extract gaussian count
#     gaussian_count = int(next(line.split()[-1] for line in header.splitlines() if line.startswith('element vertex')))
#     num_props = 62  # Total properties per Gaussian
#     print(gaussian_count)
    
#     # for debug purpose, comment the following line when doing real experiment
#     gaussian_count=gaussian_count//50

#     opacities = np.zeros(gaussian_count)
#     colors = np.zeros(3 * gaussian_count)  # 3*length
#     cov3ds = []  # 6*length
#     positions = np.zeros(3 * gaussian_count)  # 3*length
    

#     for i in range(gaussian_count):
#         offset = header_end + i * num_props * 4
#         position = struct.unpack_from('<fff', content, offset)
#         harmonic = struct.unpack_from('<fff', content, offset + 6 * 4)
#         opacity_raw = struct.unpack_from('<f', content, offset + (6 + 48) * 4)[0]
#         scale = struct.unpack_from('<fff', content, offset + (6 + 49) * 4)
#         rotation = struct.unpack_from('<ffff', content, offset + (6 + 52) * 4)
#         # # Normalize quaternion
#         # rotation = np.array(rotation) / np.linalg.norm(rotation)

#         # # Convert scale and rotation to covariance
#         # scale = np.exp(scale)

        
#         # transform rotation and scale to float32 type to match JS's Float32Array in the original code
#         rotation = np.array(rotation, dtype=np.float32)
#         scale = np.array(scale, dtype=np.float32)
#         # Normalize quaternion (manual computation to realize more similar function as JS's)
#         length2 = np.sum(rotation * rotation)
#         length = np.sqrt(length2).astype(np.float32)  # ensure that the length is float32
#         rotation = rotation / length
#         scale = np.exp(scale).astype(np.float32)
#         cov3d = compute_cov3d(scale, 1, rotation)

#         # Activate opacity
#         opacity = sigmoid(opacity_raw)

#         # Color based on harmonic
#         sh_c0 = 0.28209479177387814
#         color = [0.5 + sh_c0 * harmonic[0], 0.5 + sh_c0 * harmonic[1], 0.5 + sh_c0 * harmonic[2]]

#         # fill the opacity array
#         opacities[i] = opacity

#         # fill the color and position array
#         colors[3 * i: 3 * (i + 1)] = color
#         positions[3 * i: 3 * (i + 1)] = position
#         # fill the covariance array
#         cov3ds.extend(cov3d)
#     print("finish preprocessing")

#     def generate_batches(batch_size, gaussian_count):
#         global scene_min, scene_max  # global variable
#         # analyze the data and returned in batches
#         print(batch_size)
#         print(gaussian_count)
#         num_batches = (gaussian_count + batch_size - 1) // batch_size  # number of batches, round up
#         print(f"Total batches: {num_batches}")

#         firstTime = True
#         for batch_index in range(num_batches):  # loop to generate batches
#             if batch_index != 0:
#                 firstTime = False
#             start_index = batch_index * batch_size
#             # ensure that the ending index will not exceed the total number of Gaussians
#             end_index = min(start_index + batch_size, gaussian_count)

#             # Get the splat attributes data needed
#             opacities_batch = opacities[start_index:end_index].tolist()
#             colors_batch = colors[3 * start_index: 3 * end_index].tolist()
#             positions_batch = positions[3 * start_index: 3 * end_index]
#             cov3ds_batch = cov3ds[6 * start_index: 6 * end_index]
#             cov3ds_batch = [float(value) for value in cov3ds_batch]

#             # get the scene min and scebe max of current batch
#             # TODO: check the rendering pipeline of the usage of the scene min and scebe max
#             for position in positions_batch:
#                 scene_min = np.minimum(scene_min, position)
#                 scene_max = np.maximum(scene_max, position)
#             scene_min_batch = scene_min.tolist()
#             scene_max_batch = scene_max.tolist()
#             print('scenemin',scene_min)
#             print(scene_min_batch)

#             positions_batch = positions_batch.tolist()

#             # update count，stands for the number of points that is already sent
#             count = end_index - start_index

#             # construct data and send the data to the client through SSE
#             data = {
#                 'isFirst' : firstTime,
#                 'gaussians': {
#                     'colors': colors_batch,
#                     'cov3Ds': cov3ds_batch,
#                     'opacities': opacities_batch,
#                     'positions': positions_batch,
#                     'count': count,  # number of splats in the current batch
#                     'sceneMin': scene_min_batch,
#                     'sceneMax': scene_max_batch,
#                     'total_gs_num':gaussian_count
#                 }
#             }
            
#             yield f"data: {json.dumps(data)}\n\n"
            
#             # control the time interval of sending each request
#             time.sleep(2)

#     opacities, colors, positions, cov3ds = sort_data(opacities, colors, positions, cov3ds)
#     # assuming that the data is processed, call the function to generate the batches
#     batch_num=3
#     return Response(generate_batches(gaussian_count // batch_num, gaussian_count), content_type='text/event-stream')


@app.route('/api/load_scene', methods=['GET'])
def load_scene():
    # 读取本地 JSON 文件
    try:
        with open('brightness_rooms.json', 'r') as file:
            scene_data = json.load(file)
    except FileNotFoundError:
        return jsonify({"error": "File 'rooms.json' not found"}), 404

    opacities = np.array(scene_data['opacities'])
    colors = np.array(scene_data['colors'])
    positions = np.array(scene_data['positions'])
    cov3ds = np.array(scene_data['cov3ds'])
    gaussian_count = scene_data['gaussian_count']
    print("Finished reading data")

    def generate_batches(batch_size, gaussian_count):
        global scene_min, scene_max
        num_batches = (gaussian_count + batch_size - 1) // batch_size
        print(f"Total batches: {num_batches}")

        firstTime = True
        for batch_index in range(num_batches):
            if batch_index != 0:
                firstTime = False
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, gaussian_count)

            opacities_batch = opacities[start_index:end_index].tolist()
            colors_batch = colors[3 * start_index: 3 * end_index].tolist()
            positions_batch = positions[3 * start_index: 3 * end_index]
            cov3ds_batch = cov3ds[6 * start_index: 6 * end_index].tolist()

            # 更新场景的最小值和最大值
            for position in positions_batch:
                scene_min = np.minimum(scene_min, position)
                scene_max = np.maximum(scene_max, position)
            scene_min_batch = scene_min.tolist()
            scene_max_batch = scene_max.tolist()

            positions_batch = positions_batch.tolist()
            count = end_index - start_index

            data = {
                'gaussians': {
                    'colors': colors_batch,
                    'cov3Ds': cov3ds_batch,
                    'opacities': opacities_batch,
                    'positions': positions_batch,
                    'count': count,
                    'sceneMin': scene_min_batch,
                    'sceneMax': scene_max_batch,
                    'total_gs_num': gaussian_count
                }
            }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)

    # 分批传输数据
    batch_size = gaussian_count // 3  # 调整批量大小
    return Response(generate_batches(batch_size, gaussian_count), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)