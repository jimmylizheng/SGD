# SGD: Scalable Gaussian Delivery for 3D Scene Transmission

This is the git repo for the implementation of the CSE589 course project: SGD: Scalable Gaussian Delivery for 3D Scene Transmission. The implementation is based on the previous [3D Gaussian Splatting Web Viewer](https://github.com/kishimisu/Gaussian-Splatting-WebGL). Javascript and WebGL2 are used for the implementation of a 3D gaussian rasterizer based on the paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

## Quick Start
Clone the project to your local machine.

Download the `.ply` file containing the Gaussian splats using the following command and put it at the root directory of the project (you can use wget <URL_TO_PLY_FILE>).

Use processData.py to convert the .ply file into a .json file: "python3 processData.py" (need to change the utility functions and output directory following the comments in the code).

Run the command "python3 app.py" in your CLI to launch the server (need to change the directory of the generated .json file and server parameters following the comments in the code).

Run the command "python3 -m http.server 8000" in your CLI to launch the client.

Access the client rendering result through "http://localhost:8000/"

Follow the GUI's instruction to navigate around the scene.

## Evaluation Guide

To collect user traces, click "Start Logging" to start logging the path. Click "Save Path" to save the logged path.

Click "Load Path" to load the recorded path.

Click "Start Replay" to replay the loaded path with scalable loading and the program will automatically take screenshots for evaluation.

Run "python3 evaluation.py" to compute the evaluation result. Follow the comments in evaluation.py to modify the data directories.

Run "python3 bar_plot.py" to plot the evaluation results. Change the corresponding values in the code based on the evaluation.py's output.

Run "python3 dist_plot.py" to plot the cdf of splat attributes.

## Implementation Details

In this implementation, each gaussian is processed by a vertex shader to create a screen-space bounding rectangle made of 4 vertices, which is then colorized using a fragment shader.

#### Scale, Rotation, 3D covariance

In the original implementation, the scale and rotation attributes for each gaussian are sent to the GPU in order to calculate its 3D covariance matrix, which is ultimately used to compute its screen-space bouding rectangle. This allow to dynamically resize the splats for visualization purposes.

In this implementation, the 3D covariance is pre-computed as a one-time operation to avoid recomputing it at each frame, and also avoid sending the scale and rotation attributes to the GPU.
The splat size parameter is used differently to still allow to dynamically resize the splats 

#### Harmonics

The gaussians don't have a "color" attribute, instead their color is encoded using 16 spherical harmonics (that are vectors of 3 components). This allow for a more realistic view-dependant lighting, however it needs 48 floats per gaussian which is huge for scenes that typically have millions of gaussians.
Fortunately, not all of the harmonics coefficients are necessary to compute the final color, using more will only increase the accuracy. Here are the different degrees we can use:

- Degree 3: 16 harmonics (48 floats) per gaussian
- Degree 2: 9 harmonics (27 floats) per gaussian
- Degree 1: 4 harmonics (12 floats) per gaussian
- Degree 0: 1 harmonic (3 floats) per gaussian [no view-dependant lighting]

All degrees above 0 are view-dependant and the color for each gaussian needs to be recomputed each time the view matrix is updated.
Using degree 0 for this implementation is clearly the best in term of performances as it avoid sending any spherical harmonic to the GPU, and allow to pre-compute the gaussian color as a one-time operation before rendering.
The visual impact is clearly negligible compared to the performance gain.

## Code Structure

- **processData.py**: Processes a .ply file to generate a structured .json file for scheduler.
- **app.py**: Flask backend that serves the rendered scene and handles communication between the client 
and server.
- **dist_plot.py**: Plot cdf for different splat attributes.
- **evaluation.py**: Evaluation script that calculate average SSIM, average PSNR and QoE for multiple directories.
- **evaluation_cmp.py**: Evaluation script that calculate average SSIM and average PSNR for two directories.
- **bar_plot.py**: Produce bar plots to visualize the result.

src/
- **main.js**: Setup the main thread and do the rendering.
- **load_worker.js**: Worker thread script that receives data stream from the server and send it to the main thread for rendering.
- **loader.js**: Load and pre-process a .ply file containing gaussian data (not used by SGD).
- **worker-sort.js**: Web Worker that sorts gaussian splats by depth.
- **camera.js**: Camera manager.
- **utils.js**: WebGL & utilities.
- **gui.js**: Setup GUI.
- **gizmo.js**: gizmo render code.

shaders/
- **splat_vertex.glsl**: vertex shader that processes 4 vertices per gaussian to compute its 2d bounding quad

- **splat_fragment.glsl**: fragment shader that processes and colorize all pixels for each gaussian

## Reference

- [Original paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (3D Gaussian Splatting
for Real-Time Radiance Field Rendering)

- [Webgl Splat Renderer by antimatter15](https://github.com/antimatter15/splat): clean and concise implementation with no external library from which are coming many optimizations related to sorting (web-worker, view matrix difference treshold, count sort)

- [3D Gaussian Splatting Web Viewer](https://github.com/kishimisu/Gaussian-Splatting-WebGL): Original code base that this project based on.

## Authors
Zheng Li (jimmyli@umich.edu), Tao Wei (taowe@umich.edu), Zhengwei Wang (antwzw@umich.edu)