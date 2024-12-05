// Listen for messages from the main thread
onmessage = function (event) {
    const { scene, file } = event.data
    let allGaussians = {
        gaussians: {
            colors: [],
            cov3Ds: [],
            opacities: [],
            positions: [],
            count: 0,
            sceneMin: new Array(3).fill(Infinity),
            sceneMax: new Array(3).fill(-Infinity),
            gaussianCount: 0,
            total_gs_num: 0
        }
    };

    let partialGaussians = {
        gaussians: {
            colors: [],
            cov3Ds: [],
            opacities: [],
            positions: [],
            count: 0,
            sceneMin: new Array(3).fill(Infinity),
            sceneMax: new Array(3).fill(-Infinity),
            gaussianCount: 0,
            total_gs_num: 0
        }
    };

    // Create a StreamableReader from a URL Response object
    if (scene != null) {
        const url = `http://127.0.0.1:5000/api/load_scene?scene=${encodeURIComponent(scene)}`; // specify the port
        try {
            console.log("start to process SSE stream");
    
            // use EventSource to process SSE stream
            const eventSource = new EventSource(url);
            
            // TODO: [Violation] 'message' handler took <N>ms
            eventSource.onmessage = function(event) {                
                // parse the received data for each batch
                const responseData = JSON.parse(event.data);
                console.log(responseData); // log the returned data

                // append new data to current data
                allGaussians.gaussians.count += responseData.gaussians.count;
                allGaussians.gaussians.colors = allGaussians.gaussians.colors.concat(responseData.gaussians.colors);
                allGaussians.gaussians.cov3Ds = allGaussians.gaussians.cov3Ds.concat(responseData.gaussians.cov3Ds);
                allGaussians.gaussians.opacities = allGaussians.gaussians.opacities.concat(responseData.gaussians.opacities);
                allGaussians.gaussians.positions = allGaussians.gaussians.positions.concat(responseData.gaussians.positions);

                allGaussians.gaussians.gaussianCount = allGaussians.gaussians.count 
                // **update sceneMin and sceneMax for each batch**
                allGaussians.gaussians.sceneMin = responseData.gaussians.sceneMin
                allGaussians.gaussians.sceneMax = responseData.gaussians.sceneMax

                partialGaussians.gaussians.count = responseData.gaussians.count;
                partialGaussians.gaussians.colors = responseData.gaussians.colors;
                partialGaussians.gaussians.cov3Ds = responseData.gaussians.cov3Ds;
                partialGaussians.gaussians.opacities = responseData.gaussians.opacities;
                partialGaussians.gaussians.positions = responseData.gaussians.positions;
                // **update sceneMin and sceneMax for each batch**
                partialGaussians.gaussians.sceneMin = responseData.gaussians.sceneMin;
                partialGaussians.gaussians.sceneMax = responseData.gaussians.sceneMax;

                partialGaussians.gaussians.total_gs_num = responseData.gaussians.total_gs_num;

                // process the received 3DGS data
                // postMessage(allGaussians); // send the accumulated 3DGS data to the main thread
                postMessage(partialGaussians); // send partial 3DGS data to the main thread

                // close the event souorce
                if (responseData.gaussians.total_gs_num<=allGaussians.gaussians.gaussianCount) {
                    eventSource.close();
                    console.log("EventSource connection closed.");
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
    
        } catch (error) {
            console.error('Error loading scene:', error);
            return;
        }
    }
    else {
        throw new Error('No scene or file specified')
    }
};

