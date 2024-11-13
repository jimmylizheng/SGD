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

@app.route('/api/load_scene', methods=['GET'])
def load_scene():
    
    # get the name of the scene from the args of the request
    scene = request.args.get('scene')
    print("Msg Recd", scene)
    if not scene:
        return jsonify({"error": "Scene parameter is required"}), 400

    # construct the file name
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
    # pre allocate array
    opacities = np.zeros(gaussian_count)
    colors = np.zeros(3 * gaussian_count)  # 3*length
    cov3ds = []  # 6*length
    positions = np.zeros(3 * gaussian_count)  # 3*length
    

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

        
        # transform rotation and scale to float32 type to match JS's Float32Array in the original code
        rotation = np.array(rotation, dtype=np.float32)
        scale = np.array(scale, dtype=np.float32)
        # Normalize quaternion (manual computation to realize more similar function as JS's)
        length2 = np.sum(rotation * rotation)
        length = np.sqrt(length2).astype(np.float32)  # ensure that the length is float32
        rotation = rotation / length
        scale = np.exp(scale).astype(np.float32)


        # if i == 0:
        #     # print("First iteration - rotation:", rotation)
        #     # print("First iteration - scale (before exp):", scale)   
        #     print("First iteration - rotation: ", [f"{r:.20f}" for r in rotation])
        #     print("First iteration - scale (after exp): ",[f"{r:.20f}" for r in scale])
        #     # Compute scaling matrix
        #     S = np.diag([1 * scale[0], 1 * scale[1], 1 * scale[2]])
        #     print(f"First iteration - S0: {S[0, 0]:.20f}")
        #     print(f"First iteration - S1: {S[0, 1]:.20f}")
        #     print(f"First iteration - S2: {S[0, 2]:.20f}")
        #     print(f"First iteration - S3: {S[1, 0]:.20f}")
        #     print(f"First iteration - S4: {S[1, 1]:.20f}")
        #     print(f"First iteration - S5: {S[1, 2]:.20f}")
        #     print(f"First iteration - S6: {S[2, 0]:.20f}")
        #     print(f"First iteration - S7: {S[2, 1]:.20f}")
        #     print(f"First iteration - S8: {S[2, 2]:.20f}")

        #     # Quaternion to rotation matrix
        #     r, x, y, z = rotation
        #     print(f"First iteration - r: {r:.20f}")
        #     print(f"First iteration - x: {x:.20f}")
        #     print(f"First iteration - y: {y:.20f}")
        #     print(f"First iteration - z: {z:.20f}")
        #     R = np.array([
        #         [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
        #         [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
        #         [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)]
        #     ])
        #     print(f"First iteration - R0: {R[0, 0]:.20f}")
        #     print(f"First iteration - R1: {R[0, 1]:.20f}")
        #     print(f"First iteration - R2: {R[0, 2]:.20f}")
        #     print(f"First iteration - R3: {R[1, 0]:.20f}")
        #     print(f"First iteration - R4: {R[1, 1]:.20f}")
        #     print(f"First iteration - R5: {R[1, 2]:.20f}")
        #     print(f"First iteration - R6: {R[2, 0]:.20f}")
        #     print(f"First iteration - R7: {R[2, 1]:.20f}")
        #     print(f"First iteration - R8: {R[2, 2]:.20f}")

        #     # Compute 3D world covariance matrix Sigma
        #     # M = S @ R
        #     M = R @ S
        #     print(f"First iteration - M0: {M[0, 0]:.20f}")
        #     print(f"First iteration - M1: {M[0, 1]:.20f}")
        #     print(f"First iteration - M2: {M[0, 2]:.20f}")
        #     print(f"First iteration - M3: {M[1, 0]:.20f}")
        #     print(f"First iteration - M4: {M[1, 1]:.20f}")
        #     print(f"First iteration - M5: {M[1, 2]:.20f}")
        #     print(f"First iteration - M6: {M[2, 0]:.20f}")
        #     print(f"First iteration - M7: {M[2, 1]:.20f}")
        #     print(f"First iteration - M8: {M[2, 2]:.20f}")
        #     # Sigma = M.T @ M
        #     Sigma = M @ M.T
        #     print(f"First iteration - Sigma0: {Sigma[0, 0]:.20f}")
        #     print(f"First iteration - Sigma1: {Sigma[0, 1]:.20f}")
        #     print(f"First iteration - Sigma2: {Sigma[0, 2]:.20f}")
        #     print(f"First iteration - Sigma3: {Sigma[1, 0]:.20f}")
        #     print(f"First iteration - Sigma4: {Sigma[1, 1]:.20f}")
        #     print(f"First iteration - Sigma5: {Sigma[1, 2]:.20f}")
        #     print(f"First iteration - Sigma6: {Sigma[2, 0]:.20f}")
        #     print(f"First iteration - Sigma7: {Sigma[2, 1]:.20f}")
        #     print(f"First iteration - Sigma8: {Sigma[2, 2]:.20f}")
        cov3d = compute_cov3d(scale, 1, rotation)

        # Activate opacity
        opacity = sigmoid(opacity_raw)

        # Color based on harmonic
        sh_c0 = 0.28209479177387814
        color = [0.5 + sh_c0 * harmonic[0], 0.5 + sh_c0 * harmonic[1], 0.5 + sh_c0 * harmonic[2]]

        # fill the opacity array
        opacities[i] = opacity

        # fill the color and position array
        colors[3 * i: 3 * (i + 1)] = color
        positions[3 * i: 3 * (i + 1)] = position
        # fill the covariance array
        cov3ds.extend(cov3d)
    print("finish preprocessing")

    def generate_batches(batch_size, gaussian_count):
        global scene_min, scene_max  # global variable
        # analyze the data and returned in batches
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

            # update count，stands for the number of points that is already sent
            count = end_index - start_index

            # construct data and send the data to the client through SSE
            data = {
                'gaussians': {
                    'colors': colors_batch,
                    'cov3Ds': cov3ds_batch,
                    'opacities': opacities_batch,
                    'positions': positions_batch,
                    'count': count,  # number of splats in the current batch
                    'sceneMin': scene_min_batch,
                    'sceneMax': scene_max_batch,
                    'total_gs_num':gaussian_count
                }
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # control the time interval of sending each request
            time.sleep(2)

    # assuming that the data is processed, call the function to generate the batches
    batch_num=5
    return Response(generate_batches(gaussian_count // batch_num, gaussian_count), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
