let gl, program
let cam = null
let worker = null
let isWorkerSorting = false
let canvasSize = [0, 0]

let renderFrameRequest = null
let renderTimeout = null

let gaussianCount
let sceneMin, sceneMax

let gizmoRenderer = new GizmoRenderer()
let positionBuffer, positionData, opacityData

let allGaussians = {
    gaussians: {
        colors: [],
        cov3Ds: [],
        opacities: [],
        positions: [],
        count: 0,
    }
};


const settings = {
    scene: 'room',
    renderResolution: 0.2,
    maxGaussians: 1e6,
    scalingModifier: 1,
    sortingAlgorithm: 'count sort',
    bgColor: '#000000',
    speed: 0.07,
    fov: 47,
    debugDepth: false,
    freeFly: false,
    sortTime: 'NaN',
    uploadFile: () => document.querySelector('#input').click(),

    // Camera calibration
    calibrateCamera: () => {},
    finishCalibration: () => {},
    showGizmo: true,

    // Camera path following
    // startPathFollow: () => cam.startPathFollow(), // Starts camera movement along path
    // stopPathFollow: () => cam.stopPathFollow(),  // Stops camera movement along path
    // pathDuration: 10, // Default duration (in seconds)
    // setPath: () => {} // Placeholder function, updated later
    startPathLogging: () => cam.startPathLogging(100), // Start logging with an interval of 100ms
    // stopPathLogging: () => stopPathLogging(),     // Stop logging
    savePath: () => cam.saveLoggedPath(),         // Save the logged path
    loadPath: () => {},                           // Placeholder for loading
    startReplay: () => cam.startPathReplay(),     // Start replaying the path
    pathDuration: 10                              // Default path duration for replay
}

const defaultCameraParameters = {
    'room': {
        up: [0, 0.886994, 0.461779],
        target: [-0.428322434425354, 1.2004123210906982, 0.8184626698493958],
        camera: [4.950796326794864, 1.7307963267948987, 2.5],
        defaultCameraMode: 'freefly',
        size: '270mb'
    },
    'building': {
        up: [0, 0.968912, 0.247403],
        target: [-0.262075, 0.76138, 1.27392],
        camera: [ -1.1807959999999995, 1.8300000000000007, 3.99],
        defaultCameraMode: 'orbit',
        size: '326mb'
    },
    'garden': {
        up: [0.055540, 0.928368, 0.367486],
        target: [0.338164, 1.198655, 0.455374],
        defaultCameraMode: 'orbit',
        size: '1.07gb [!]'
    }
}

// When capture is true, capture a screenshot
let capture = false;
// Keypress listener, press c to capture the screenshot
document.addEventListener('keydown', (event) => {
    // Check if the 'c' key is pressed
    if (event.key === 'c' || event.key === 'C') {
        console.log('C key pressed: Taking screenshot...');
        // takeScreenshot();
        capture = true;
    }
});

// Variables and functons for path logging function
// let logInterval = null;

// function startPathLogging(interval = 100) {
//     console.log("executing startPathLogging")
//     if (logInterval) return; // Avoid duplicate intervals
    // logInterval = setInterval(() => {
    //     cam.logCurrentPosition();
    // }, interval);
// }

// function stopPathLogging() {
//     console.log("executing stopPathLogging")
//     if (logInterval) {
//         clearInterval(logInterval);
//         logInterval = null;
//     }
// }

async function main() {
    // Setup webgl context and buffers
    const { glContext, glProgram, buffers } = await setupWebglContext()
    gl = glContext; program = glProgram // Handy global vars

    if (gl == null || program == null) {
        document.querySelector('#loading-text').style.color = `red`
        document.querySelector('#loading-text').textContent = `Could not initialize the WebGL context.`
        throw new Error('Could not initialize WebGL')
    }

    // Setup web worker for multi-threaded sorting
    worker = new Worker('src/worker-sort.js')

    // Event that receives sorted gaussian data from the worker
    worker.onmessage = e => {
        const { data, sortTime } = e.data
        console.log("worker rcvd msg");
        if (sortTime==0){
            cam.disableMovement = false
        }

        if (getComputedStyle(document.querySelector('#loading-container')).opacity != 0) {
            document.querySelector('#loading-container').style.opacity = 0
            cam.disableMovement = false
        }

        const updateBuffer = (buffer, data) => {
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
            gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW)
        }

        updateBuffer(buffers.color, data.colors)
        updateBuffer(buffers.center, data.positions)
        updateBuffer(buffers.opacity, data.opacities)
        updateBuffer(buffers.covA, data.cov3Da)
        updateBuffer(buffers.covB, data.cov3Db)

        // Needed for the gizmo renderer
        positionBuffer = buffers.center
        positionData = data.positions
        opacityData = data.opacities

        settings.sortTime = sortTime

        isWorkerSorting = false
        requestRender()
    }

    // Setup GUI
    initGUI()

    // Setup gizmo renderer
    await gizmoRenderer.init()

    // Load the default scene (for real application)
    await loadScene({ scene: settings.scene })

    // Load the default scene and do the path replay at the same time (for measurement)
    // const loadScenePromise = loadScene({ scene: settings.scene });
    // const LoadandReplayPromise = loadandReplayPath();
    // await Promise.all([loadScenePromise, LoadandReplayPromise]);

    console.log("main ends");
}

// load and replay the path
// async function loadandReplayPath() {
//     try {
//         const fileInput = document.getElementById("fileInput");
//         cam.loadPathFromFile(filePath);

//         // const response = await fetch(filePath);
//         // console.log("get response successfully");
//         // // const response = await fetch("./pathfile.json");

//         // if (!response.ok) throw new Error(`Failed to load path file: ${response.statusText}`);

//         // const path = await response.json();
//         // if (!Array.isArray(path) || path.length === 0) throw new Error("Invalid path format");

//         // cam.loggedPath = path;
//         console.log("Path loaded successfully");
//         cam.startPathReplay();
//     } catch (error) {
//         console.error("Error loading path on page load:", error);
//         // Optionally load a default path or show a message
//         // cam.loggedPath = getDefaultPath(); // Example: Provide a fallback path
//         // cam.startPathReplay();
//     }
// }

// Load a .ply scene specified as a name (URL fetch) or local file
async function loadScene({scene, file}) {
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    if (cam) cam.disableMovement = true
    document.querySelector('#loading-container').style.opacity = 1

    
    sceneMin = new Array(3).fill(Infinity)
    sceneMax = new Array(3).fill(-Infinity)

    // Create a StreamableReader from a URL Response object
    if (scene != null) {
        console.log(scene);
        scene = scene.split('(')[0].trim();
        console.log(scene);
        const url = `http://127.0.0.1:5000/api/load_scene?scene=${encodeURIComponent(scene)}`; // specify the port
        try {
            if (cam && cam.LoadEnd) {
                // clear the previous data
                cam.LoadEnd=false;
                // gaussians: {
                //     colors: [],
                //     cov3Ds: [],
                //     opacities: [],
                //     positions: [],
                //     count: 0,
                // }
                allGaussians.gaussians.count = 0;
                allGaussians.gaussians.colors = [];
                allGaussians.gaussians.cov3Ds = [];
                allGaussians.gaussians.opacities = [];
                allGaussians.gaussians.positions = [];
                console.log("clear the previous data and reload the scene");
            }
            console.log("start to process SSE stream");

            const cameraParameters = scene ? defaultCameraParameters[scene] : {}
            if (cam == null) cam = new Camera(cameraParameters)
    
            // use EventSource to process SSE stream
            const eventSource = new EventSource(url);

            // start replaying when the client is receiving the data
            // cam.startPathReplay();
            
            // TODO: [Violation] 'message' handler took <N>ms
            eventSource.onmessage = function(event) {
                // parse the received data for each batch
                const responseData = JSON.parse(event.data);
                console.log(responseData); // log the returned data

                // if (responseData.isFirst && !this.hasStartedReplay) {
                //     console.log("First data received, starting path replay...");
                //     this.hasStartedReplay = true; // 防止重复触发
                //     cam.startPathReplay(); // 调用异步方法
                // }
    
                // append new data to current data
                allGaussians.gaussians.count += responseData.gaussians.count;
                allGaussians.gaussians.colors = allGaussians.gaussians.colors.concat(responseData.gaussians.colors);
                allGaussians.gaussians.cov3Ds = allGaussians.gaussians.cov3Ds.concat(responseData.gaussians.cov3Ds);
                allGaussians.gaussians.opacities = allGaussians.gaussians.opacities.concat(responseData.gaussians.opacities);
                allGaussians.gaussians.positions = allGaussians.gaussians.positions.concat(responseData.gaussians.positions);

                gaussianCount = allGaussians.gaussians.count // ?? question here
                 // **update sceneMin and sceneMax for each batch**
                 sceneMin = responseData.gaussians.sceneMin
                 sceneMax = responseData.gaussians.sceneMax

                // debug info
                console.log("Updated sceneMin:", sceneMin);
                console.log("Updated sceneMax:", sceneMax);

                // console.log("debug position:", allGaussians.gaussians.positions[0]);
                // console.log("debug opacities:", allGaussians.gaussians.opacities[0]);
                // console.log("debug colors:", allGaussians.gaussians.colors[0]);
                // console.log("debug cov3Ds:", allGaussians.gaussians.cov3Ds[0]);
                
                // hide the loading icon when starting receving information
                if (getComputedStyle(document.querySelector('#loading-container')).opacity != 0) {
                    document.querySelector('#loading-container').style.opacity = 0
                    cam.disableMovement = false
                }

                // process the received 3DGS data
                worker.postMessage(allGaussians); // send the accumulated 3DGS data to Web Worker
                // const cameraParameters = scene ? defaultCameraParameters[scene] : {}
                // if (cam == null) cam = new Camera(cameraParameters)
                // if (responseData.isFirst && !this.hasStartedReplay) {
                //     console.log("First data received, starting path replay...");
                //     this.hasStartedReplay = true; // 防止重复触发
                //     cam.startPathReplay(); // 调用异步方法
                // }

                // else cam.setParameters(cameraParameters)
                cam.update()

                // Update GUI
                // settings.maxGaussians controls the max number of splats to be rendered
                if(settings.maxGaussians>gaussianCount){
                    settings.maxGaussians = Math.min(settings.maxGaussians, gaussianCount)
                }
                else{
                    settings.maxGaussians = Math.max(settings.maxGaussians, gaussianCount)
                }
                maxGaussianController.max(gaussianCount)
                maxGaussianController.updateDisplay()

                // close the event souorce
                if (responseData.gaussians.total_gs_num<=gaussianCount) {
                    cam.LoadEnd=true;
                    eventSource.close();
                    console.log("EventSource connection closed.");
                    console.log("cam.disableMovement: ",cam.disableMovement);
                    // capture the screenshot
                    setTimeout(() => {
                        console.log('call the function to capture the screenshot');
                        capture=true;
                    }, 2000);
                }
            };
    
            // eventSource.onerror = function(event) {
            //     // deal with the error situation
            //     console.error('Error occurred in SSE stream:', event);
            //     eventSource.close();
            // };
    
            eventSource.onopen = function() {
                console.log('SSE connection established');
            };

            // Setup camera
    
        } catch (error) {
            console.error('Error loading scene:', error);
            return;
        }
    }
     else if (file != null) {
        contentLength = file.size
        reader = file.stream().getReader()
        settings.scene = 'custom'
    } else {
        throw new Error('No scene or file specified')
    }
    // Send gaussian data to the worker
    
    // worker.postMessage({ gaussians: {
    //     ...data, count: gaussianCount
    // } })
    

}

function requestRender(...params) {
    if (renderFrameRequest != null) 
        cancelAnimationFrame(renderFrameRequest)

    renderFrameRequest = requestAnimationFrame(() => render(...params)) 
}

// Render a frame on the canvas
function render(width, height, res) {
    // Update canvas size
    const resolution = res ?? settings.renderResolution
    const canvasWidth = width ?? Math.round(canvasSize[0] * resolution)
    const canvasHeight = height ?? Math.round(canvasSize[1] * resolution)

    if (gl.canvas.width != canvasWidth || gl.canvas.height != canvasHeight) {
        gl.canvas.width = canvasWidth
        gl.canvas.height = canvasHeight
    }

    // Setup viewport
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.useProgram(program)


    // Update camera path-following and position (for path following function)
    // cam.updatePathFollow();
    // logCurrentPosition();
    cam.updatePathReplay();
    // Update camera
    cam.update()

    // Original implementation parameters
    const W = gl.canvas.width
    const H = gl.canvas.height
    const tan_fovy = Math.tan(cam.fov_y * 0.5)
    const tan_fovx = tan_fovy * W / H
    const focal_y = H / (2 * tan_fovy)
    const focal_x = W / (2 * tan_fovx)

    gl.uniform1f(gl.getUniformLocation(program, 'W'), W)
    gl.uniform1f(gl.getUniformLocation(program, 'H'), H)
    gl.uniform1f(gl.getUniformLocation(program, 'focal_x'), focal_x)
    gl.uniform1f(gl.getUniformLocation(program, 'focal_y'), focal_y)
    gl.uniform1f(gl.getUniformLocation(program, 'tan_fovx'), tan_fovx)
    gl.uniform1f(gl.getUniformLocation(program, 'tan_fovy'), tan_fovy)
    gl.uniform1f(gl.getUniformLocation(program, 'scale_modifier'), settings.scalingModifier)
    // gl.uniform3fv(gl.getUniformLocation(program, 'boxmin'), sceneMin)
    // gl.uniform3fv(gl.getUniformLocation(program, 'boxmax'), sceneMax)
    gl.uniform3fv(gl.getUniformLocation(program, 'boxmin'), new Float32Array(sceneMin));
    gl.uniform3fv(gl.getUniformLocation(program, 'boxmax'), new Float32Array(sceneMax));

    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'projmatrix'), false, cam.vpm)
    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'viewmatrix'), false, cam.vm)

    // Custom parameters
    gl.uniform1i(gl.getUniformLocation(program, 'show_depth_map'), settings.debugDepth)

    // Draw
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, settings.maxGaussians)

    // Draw gizmo
    gizmoRenderer.render()

    if(cam.capture){
        capture=true;
        cam.capture=false;
        console.log("cam capture is true")
    }
    if (capture) {
        capture=false;
        if(cam.pic_list.length){
            takeScreenshot(cam.pic_list[cam.pic_list.length - 1]);
        }
        else{takeScreenshot();}
    }

    renderFrameRequest = null

    // Progressively draw with higher resolution after the camera stops moving
    let nextResolution = Math.floor(resolution * 4 + 1) / 4
    if (nextResolution - resolution < 0.1) nextResolution += .25

    if (nextResolution <= 1 && !cam.needsWorkerUpdate && !isWorkerSorting) {
        const nextWidth = Math.round(canvasSize[0] * nextResolution)
        const nextHeight = Math.round(canvasSize[1] * nextResolution)

        if (renderTimeout != null) 
            clearTimeout(renderTimeout)

        renderTimeout = setTimeout(() => requestRender(nextWidth, nextHeight, nextResolution), 200)
    }
}

window.onload = main