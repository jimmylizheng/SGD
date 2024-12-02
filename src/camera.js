const { mat4, vec3, vec4 } = glMatrix

class Camera {
    constructor({target = [0, 0, 0], up = [0, 1, 0], camera = [], defaultCameraMode} = {}) {
        this.target = [...target] // Position of look-at target
        this.up = [...up]         // Up vector

        // Camera spherical coordinates (around the target)
        this.theta  = camera[0] ?? -Math.PI/2
        this.phi    = camera[1] ?? Math.PI/2
        this.radius = camera[2] ?? 3

        // Y Field of view
        this.fov_y = 0.820176

        // False: orbit around object (mouse + wheel)
        // True: free-fly (mouse + AWSD)
        this.freeFly = settings.freeFly = defaultCameraMode !== 'orbit'

        // Indicate that the camera moved and the splats need to be sorted
        this.needsWorkerUpdate = true

        // Is the user dragging the mouse?
        this.isDragging = false

        // Disable user input
        this.disableMovement = false

        // Enable ray cast for camera calibration
        this.isCalibrating = false
        this.calibrationPoints = []

        // Keyboard state
        this.keyStates = {
            KeyW: false,
            KeyS: false,
            KeyA: false,
            KeyD: false,
            ShiftLeft: false,
            Space: false
        }

        // Helper vectors
        this.pos = vec3.create()
        this.front = vec3.create()
        this.right = vec3.create()        

        // Helper matrices
        this.viewMatrix = mat4.create()
        this.projMatrix = mat4.create()
        this.viewProjMatrix = mat4.create()
        this.lastViewProjMatrix = mat4.create()
        this.sceneRotationMatrix = rotateAlign(this.up, [0, 1, 0])

        // Matrices sent to the GPU
        this.vm = mat4.create()
        this.vpm = mat4.create()

        // Rotate camera around target (mouse)
        gl.canvas.addEventListener('mousemove', e => {
            if (!e.buttons || this.disableMovement) return

            this.theta -= e.movementX * 0.01 * .5
            this.phi = Math.max(1e-6, Math.min(Math.PI - 1e-6, this.phi + e.movementY * 0.01 * .5))
            this.isDragging = true

            requestRender()
        })

        // Rotate camera around target (touch)
        const lastTouch = {}
        gl.canvas.addEventListener('touchstart', e => {
            e.preventDefault()
            if (e.touches.length == 0 || this.disableMovement) return

            lastTouch.clientX = e.touches[0].clientX
            lastTouch.clientY = e.touches[0].clientY
        })
        gl.canvas.addEventListener('touchmove', e => {
            e.preventDefault()
            if (e.touches.length == 0 || this.disableMovement) return

            const touch = e.touches[0]
            const movementX = touch.clientX - lastTouch.clientX
            const movementY = touch.clientY - lastTouch.clientY
            lastTouch.clientX = touch.clientX
            lastTouch.clientY = touch.clientY

            this.theta -= movementX * 0.01 * .5 * .3
            this.phi = Math.max(1e-6, Math.min(Math.PI - 1e-6, this.phi + movementY * 0.01 * .5))

            requestRender()
        })

        // Zoom in and out
        gl.canvas.addEventListener('wheel', e => {
            if (this.freeFly || this.disableMovement) return

            this.radius = Math.max(1, this.radius + e.deltaY * 0.01)

            requestRender()
        })

        // Free-fly movement
        document.addEventListener('keydown', e => {
            if (!this.freeFly || this.disableMovement || this.keyStates[e.code] == null) 
                return
            this.keyStates[e.code] = true
        })

        document.addEventListener('keyup', e => {
            if (!this.freeFly || this.disableMovement || this.keyStates[e.code] == null) 
                return
            this.keyStates[e.code] = false
        })

        // Gizmo event
        gl.canvas.addEventListener('mouseup', e => {
            if (this.isDragging) {
                this.isDragging = false
                return
            }
            if (this.disableMovement || !this.isCalibrating) return

            this.raycast(e.clientX, e.clientY)
        })

        // Update camera from mouse and keyboard inputs
        setInterval(this.updateKeys.bind(this), 1000/60)

        // Variables for path logging
        this.loggedPath = []; // Store the logged path
        this.pathStartTime = null; // Start time for logging
        this.pathEndTime = null; // End time for logging
        this.logInterval = null;

        // Variables for replaying and reloading
        this.LoadEnd=false; // true when the load finishes

        // Variables to do the screenshot outputs
        this.pic_list=[]; // list consists of timestamps that have been captured
        this.capture=false;

        // Variables for replaying
        this.is_first_load=true;
    }

    // Reset parameters on new scene load
    setParameters({target = [0, 0, 0], up = [0, 1, 0], camera = [], defaultCameraMode} = {}) {
        this.target = [...target]
        this.up = [...up]
        this.theta  = camera[0] ?? -Math.PI/2
        this.phi    = camera[1] ?? Math.PI/2
        this.radius = camera[2] ?? 3
        this.freeFly = settings.freeFly = defaultCameraMode !== 'orbit'
        this.needsWorkerUpdate = true
        this.sceneRotationMatrix = rotateAlign(this.up, [0, 1, 0])
        camController.resetCalibration()
    }

    updateKeys() {
        if (Object.values(this.keyStates).every(s => !s) || this.disableMovement) return

        const front = this.getFront()
        const right = vec3.cross(this.right, front, this.up)

        if (this.keyStates.KeyW) vec3.add(this.target, this.target, vec3.scale(front, front, settings.speed))
        if (this.keyStates.KeyS) vec3.subtract(this.target, this.target, vec3.scale(front, front, settings.speed))
        if (this.keyStates.KeyA) vec3.add(this.target, this.target, vec3.scale(right, right, settings.speed))
        if (this.keyStates.KeyD) vec3.subtract(this.target, this.target, vec3.scale(right, right, settings.speed))
        if (this.keyStates.ShiftLeft) vec3.add(this.target, this.target, vec3.scale(vec3.create(), this.up, settings.speed))
        if (this.keyStates.Space) vec3.subtract(this.target, this.target, vec3.scale(vec3.create(), this.up, settings.speed))

        requestRender()
    }

    getPos(radius = this.radius) {
        const pos = [
            radius * Math.sin(this.phi) * Math.cos(this.theta),
            radius * Math.cos(this.phi),
            radius * Math.sin(this.phi) * Math.sin(this.theta)
        ]

        return vec3.transformMat3(pos, pos, this.sceneRotationMatrix)
    }

    getFront() {
        const front = vec3.subtract(this.front, [0,0,0], this.getPos())
        vec3.normalize(front, front)
        return front
    }

    update() {
        // Update current position
        vec3.add(this.pos, this.target, this.getPos(this.freeFly ? 1 : this.radius))

        // Create a lookAt view matrix
        mat4.lookAt(this.viewMatrix, this.pos, this.target, this.up)

        // Create a perspective projection matrix
        const aspect = gl.canvas.width / gl.canvas.height
        mat4.perspective(this.projMatrix, this.fov_y, aspect, 0.1, 100)

		// Convert view and projection to target coordinate system
        // Original C++ reference: https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_union/src/projects/gaussianviewer/renderer/GaussianView.cpp#L464
        mat4.copy(this.vm, this.viewMatrix)
        mat4.multiply(this.vpm, this.projMatrix, this.viewMatrix)

        invertRow(this.vm, 1)
        invertRow(this.vm, 2)
        invertRow(this.vpm, 1)

        // (Webgl-specific) Invert x-axis
        invertRow(this.vm, 0)
        invertRow(this.vpm, 0)

        this.updateWorker()
    }

    updateWorker() {
        // Calculate the dot product between last and current view-projection matrices
        // If they differ too much, the splats need to be sorted
        const dot = this.lastViewProjMatrix[2]  * this.vpm[2] 
                  + this.lastViewProjMatrix[6]  * this.vpm[6]
                  + this.lastViewProjMatrix[10] * this.vpm[10]
        if (Math.abs(dot - 1) > 0.01) {
            this.needsWorkerUpdate = true
            mat4.copy(this.lastViewProjMatrix, this.vpm)
        }

        // Sort the splats as soon as the worker is available
        if (this.needsWorkerUpdate && !isWorkerSorting) {
            this.needsWorkerUpdate = false
            isWorkerSorting = true
            worker.postMessage({
                viewMatrix:  this.vpm, 
                maxGaussians: settings.maxGaussians,
                sortingAlgorithm: settings.sortingAlgorithm
            })
        }
    }

    raycast(x, y) {
        if (this.calibrationPoints.length >= 3) return
        
        // Calculate ray direction from mouse position
        const Px = (x / window.innerWidth * 2 - 1)
        const Py = -(y / window.innerHeight * 2 - 1)
        const cameraToWorld = mat4.invert(mat4.create(), this.vpm)
        const rayOriginWorld = vec3.transformMat4(vec3.create(), this.pos, cameraToWorld)
        const rayPWorld = vec3.transformMat4(vec3.create(), [Px, Py, 1], cameraToWorld)
        const rd = vec3.subtract(vec3.create(), rayPWorld, rayOriginWorld)
        vec3.normalize(rd, rd)

        // Raycast the gaussian splats
        const hit = { id: -1, dist: 1e9 }
        for (let i = 0; i < gaussianCount; i++) {
            const pos = positionData.slice(i * 3, i * 3 + 3)
            const alpha = opacityData[i]

            if (alpha < 0.1) continue

            const t = raySphereIntersection(this.pos, rd, pos, 0.1)

            if (t > 0.4 && t < hit.dist) {
                hit.id = i
                hit.dist = t
            }
        }

        if (hit.id == -1) return null

        const hitPosition = vec3.add(vec3.create(), this.pos, vec3.scale(vec3.create(), rd, hit.dist))
        this.calibrationPoints.push(hitPosition)

        // Update gizmo renderer
        gizmoRenderer.setPlaneVertices(...this.calibrationPoints)
        requestRender()

        return rd
    }

    resetCalibration() {
        this.isCalibrating = false
        this.calibrationPoints = []
        gizmoRenderer.setPlaneVertices()
    }

    finishCalibration() {
        this.isCalibrating = false
        this.calibrationPoints = []
        cam.up = gizmoRenderer.planeNormal
        cam.sceneRotationMatrix = rotateAlign(gizmoRenderer.planeNormal, [0, 1, 0])
        requestRender()
    }

    // functions for implementing path following function
    // setPath(path, duration) {
    //     console.log('execute setPath()');
    //     this.path = path; // Array of waypoints with position and target
    //     this.pathDuration = duration; // Total time to complete the path
    //     this.pathStartTime = null;
    //     this.isFollowingPath = false;
    // }

    // startPathFollow() {
    //     console.log('execute startPathFollow()');
    //     if (!this.path || this.path.length < 2) {
    //         console.error("Path is not defined or too short!");
    //         return;
    //     }
    //     this.pathStartTime = performance.now();
    //     this.isFollowingPath = true;
    // }

    // stopPathFollow() {
    //     console.log('execute stopPathFollow()');
    //     this.isFollowingPath = false;
    // }

    async startPathReplay_helper() {
        this.pic_list=[]; 
        if (!this.loggedPath || this.loggedPath.length < 2) {
            console.error("No valid path to replay!");
            return;
        }
        settings.renderResolution=1;
        this.pathStartTime = performance.now(); // Start time for replay
        this.isReplayingPath = true;
        const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        // const replay = () => {
        const replay = async () => {
            if (!this.isReplayingPath) return; // Stop if replay is interrupted
    
            // Call the existing updatePathReplay function
            // this.updatePathReplay();
    
            // Trigger rendering
            requestRender();
            await delay(20);
    
            // Schedule the next frame
            requestAnimationFrame(replay);
        };
        // Start the replay loop
        requestAnimationFrame(replay);
    }



    async startPathReplay() {
        // startPathReplay_helper();
        // this.isReplayingPath=true;
        // clear the gl buffer
        // gl.clearColor(0, 0, 0, 0);
        // gl.clear(gl.COLOR_BUFFER_BIT);
        console.log("start replay");
        gl.clearColor(0, 0, 0, 0)
        // gl.clear(gl.COLOR_BUFFER_BIT)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); // Clear color and depth buffers
        // Reset Gaussian data
        allGaussians.gaussians.count = 0;
        allGaussians.gaussians.colors = [];
        allGaussians.gaussians.cov3Ds = [];
        allGaussians.gaussians.opacities = [];
        allGaussians.gaussians.positions = [];
        worker.postMessage(allGaussians);
        await gizmoRenderer.init()
        console.log("gizmo finish init")
    
        // Request a render after clearing the scene
        requestAnimationFrame(() => render()) 
        requestRender(); // Ensure the cleared scene is rendered immediately
        // const clear_buf = () => {
        //     // clear the gl buffer
        //     gl.clearColor(0, 0, 0, 0);
        //     gl.clear(gl.COLOR_BUFFER_BIT);
    
        //     // Trigger rendering
        //     requestRender();
        // };
        // Start the replay loop
        // requestAnimationFrame(clear_buf);
        // requestRender()
        console.log("clear the previous data and reload the scene");
        // Run both functions concurrently and wait for both to complete
        // await sleep(2000); // Sleep for 2 seconds
        await Promise.all([loadScene({ scene: settings.scene }), this.startPathReplay_helper()]);
        // await Promise.all([loadScene({ scene: settings.scene })]);
    }

    updatePathReplay() {
        if (!this.isReplayingPath) return;
        // console.log("update replay");
        const elapsed = (performance.now() - this.pathStartTime) / 1000; // Time in seconds
        const path = this.loggedPath;

        // check whether or not to take a screenshot
        // console.log("checking elapsed")
        if (!this.pic_list.includes(Math.floor(elapsed))){
            console.log("log capture")
            this.pic_list.push(Math.floor(elapsed));
            this.capture=true;
        }

        // Find the segment based on time
        let segmentIndex = path.findIndex((p) => p.time > elapsed);
        if (segmentIndex === -1) {
            this.isReplayingPath = false; // Replay finished
            return;
        }

        if (segmentIndex === 0) segmentIndex = 1; // Ensure we don't use the first point

        const prev = path[segmentIndex - 1];
        const next = path[segmentIndex];

        // Interpolate between the two points
        const segmentT = (elapsed - prev.time) / (next.time - prev.time);

        const currentPosition = [
            prev.position[0] + (next.position[0] - prev.position[0]) * segmentT,
            prev.position[1] + (next.position[1] - prev.position[1]) * segmentT,
            prev.position[2] + (next.position[2] - prev.position[2]) * segmentT,
        ];

        const currentTarget = [
            prev.target[0] + (next.target[0] - prev.target[0]) * segmentT,
            prev.target[1] + (next.target[1] - prev.target[1]) * segmentT,
            prev.target[2] + (next.target[2] - prev.target[2]) * segmentT,
        ];

        // Interpolate spherical coordinates
        const currentTheta = prev.theta + (next.theta - prev.theta) * segmentT;
        const currentPhi = prev.phi + (next.phi - prev.phi) * segmentT;
        const currentRadius = prev.radius + (next.radius - prev.radius) * segmentT;

        // Update camera parameters
        this.theta = currentTheta;
        this.phi = currentPhi;
        this.radius = currentRadius;
        this.target = currentTarget;
        vec3.copy(this.pos, currentPosition);
        this.update();
    }

    // updatePathFollow() {
    //     // console.log('execute updatePathFollow()');
    //     if (!this.isFollowingPath) return;

    //     // performance.now(): Unit: Milliseconds (ms)
    //     const elapsed = (performance.now() - this.pathStartTime) / 1000; // Time in seconds
    //     const t = elapsed / this.pathDuration; // Normalized time [0, 1]

    //     if (t >= 1) {
    //         this.isFollowingPath = false; // Stop when the path is complete
    //         return;
    //     }

    //     // Determine which segment of the path we're in
    //     const segmentCount = this.path.length - 1;
    //     const segmentIndex = Math.min(Math.floor(t * segmentCount), segmentCount - 1);
    //     const segmentT = (t * segmentCount) - segmentIndex;

    //     const start = this.path[segmentIndex];
    //     const end = this.path[segmentIndex + 1];

    //     // Interpolate position
    //     const currentPosition = [
    //         start.position[0] + (end.position[0] - start.position[0]) * segmentT,
    //         start.position[1] + (end.position[1] - start.position[1]) * segmentT,
    //         start.position[2] + (end.position[2] - start.position[2]) * segmentT,
    //     ];

    //     // Interpolate target
    //     const currentTarget = [
    //         start.target[0] + (end.target[0] - start.target[0]) * segmentT,
    //         start.target[1] + (end.target[1] - start.target[1]) * segmentT,
    //         start.target[2] + (end.target[2] - start.target[2]) * segmentT,
    //     ];

    //     // Update camera parameters
    //     this.target = currentTarget;
    //     vec3.copy(this.pos, currentPosition); // Camera position
    //     this.update();
    // }

    // Functions for implementing path logging function

    // startPathLogging(interval = 100) {
    //     console.log("executing startPathLogging")
    //     this.loggedPath = [];
    //     this.pathStartTime = performance.now(); // Record the start time
    //     this.pathEndTime = null;
    //     if (logInterval) return; // Avoid duplicate intervals
    //     logInterval = setInterval(() => {
    //         cam.logCurrentPosition();
    //     }, interval);
    // }

    startPathLogging(interval = 100) {
        console.log("executing startPathLogging");
        this.loggedPath = [];
        this.pathStartTime = performance.now(); // Record the start time
        if (this.logInterval) return; // Avoid duplicate intervals
        this.logInterval = setInterval(() => {
            cam.logCurrentPosition();
        }, interval);
    }

    // stopPathLogging() {
    //     console.log("executing stopPathLogging")
    //     this.pathEndTime = performance.now(); // Record the end time
    //     if (logInterval) {
    //         clearInterval(logInterval);
    //         logInterval = null;
    //     }
    // }

    logCurrentPosition() {
        // const currentPosition = this.getPos();
        // const currentTarget = [...this.target];
        // this.loggedPath.push({ position: currentPosition, target: currentTarget });

        if (!this.pathStartTime) {
            console.error("Path logging has not started!");
            return;
        }

        const currentPosition = this.getPos();
        const currentTarget = [...this.target];
        const elapsedTime = (performance.now() - this.pathStartTime) / 1000; // Time in seconds

        this.loggedPath.push({
            position: currentPosition,
            target: currentTarget,
            theta: this.theta, // Horizontal rotation angle
            phi: this.phi, // Vertical rotation angle
            radius: this.radius, // Distance to target
            time: elapsedTime // Log the elapsed time
        });
    }

    saveLoggedPath(filename = 'logged_path.json') {
        console.log("executing saveLoggedPath");
        if (this.logInterval) {
            clearInterval(this.logInterval);
            this.logInterval = null;
        }
        if (this.loggedPath.length === 0) {
            console.warn("No path to save.");
            return;
        }
        const data = JSON.stringify(this.loggedPath, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        console.log("Path saved to file:", filename);
    }

    async loadPathFromFile(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const path = JSON.parse(event.target.result);
                if (!Array.isArray(path) || path.length === 0) {
                    throw new Error("Invalid path format");
                }
                // this.setPath(path, settings.pathDuration);
                this.loggedPath = path;
                console.log("Path loaded:", path);
            } catch (error) {
                console.error("Failed to load path:", error);
            }
        };
        reader.readAsText(file);
    }
}

const invertRow = (mat, row) => {
    mat[row + 0] = -mat[row + 0]
    mat[row + 4] = -mat[row + 4]
    mat[row + 8] = -mat[row + 8]
    mat[row + 12] = -mat[row + 12]
}

// Calculate the rotation matrix that aligns v1 with v2
// https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724
function rotateAlign(v1, v2) {
    const axis = [
      v1[1] * v2[2] - v1[2] * v2[1],
      v1[2] * v2[0] - v1[0] * v2[2],
      v1[0] * v2[1] - v1[1] * v2[0]
    ]

    const cosA = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    const k = 1.0 / (1.0 + cosA)
  
    const result = [
      (axis[0] * axis[0] * k) + cosA, (axis[1] * axis[0] * k) - axis[2], (axis[2] * axis[0] * k) + axis[1],
      (axis[0] * axis[1] * k) + axis[2], (axis[1] * axis[1] * k) + cosA, (axis[2] * axis[1] * k) - axis[0],
      (axis[0] * axis[2] * k) - axis[1], (axis[1] * axis[2] * k) + axis[0], (axis[2] * axis[2] * k) + cosA
    ]
  
    return result
}

// Calculate the intersection distance between a ray and a sphere
// Return -1 if no intersection
const raySphereIntersection = (rayOrigin, rayDirection, sphereCenter, sphereRadius) => {
    const oc = vec3.subtract(vec3.create(), rayOrigin, sphereCenter)
    const a = vec3.dot(rayDirection, rayDirection)
    const b = 2 * vec3.dot(oc, rayDirection)
    const c = vec3.dot(oc, oc) - sphereRadius * sphereRadius
    const discriminant = b * b - 4 * a * c
    if (discriminant < 0) return -1
    return (-b - Math.sqrt(discriminant)) / (2 * a)
}