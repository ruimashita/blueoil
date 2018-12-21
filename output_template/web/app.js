// Ready for init blueoil
let nn
let inputShape

// Init app
const video = document.getElementById('video')
const inference = document.getElementById('inference')
const canvas = document.getElementById('canvas')
const inferenceCtx = inference.getContext('2d')
const canvasCtx = canvas.getContext('2d')
// const inferenceCopy = document.getElementById("inferencecopy");
// const copyContext = inferenceCopy.getContext('2d');

const videoWidth = video.offsetWidth
const videoHeight = video.offsetHeight

const canvas_size = 480;
inference.width  = 160;
inference.height = 160;
inference.setAttribute("width", 160);
inference.setAttribute("height", 160);

canvas.width  = canvas_size;
canvas.height = canvas_size;


const params = {
    format_yolo_v2: {
        image_size: [160, 160],
        num_classes: 1,
        anchors: [[0.25, 0.25], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]],
        boxes_per_cell: 5,
        data_format: "NHWC",
        threshold: 0.2,
    },
    classes: ["face"],
    iou_threshold: 0.5,
}


const format_yolo_v2 = new FormatYoloV2(...Object.values(params.format_yolo_v2))
const nms = new NMS(params.classes, params.iou_threshold)
var counter = 0;

const update = () => {
    // reuslt boxes.
    let boxes = new Array()

    // Get image data
    const imageData = inferenceCtx.getImageData(0, 0, 160, 160)
    let start = performance.now();

    let input_size = inputShape.reduce((x, y) => {return x*y})
    let input = new Float32Array(input_size)

    // divid255
    j = 0;
    for (var i = 0; i < imageData.data.length; i++) {
        input[j] = parseFloat(imageData.data[i]) / 255.0;
        if ((i % 4) != 0 ){
            j++;
        }
    }

    // run network
    let nn_start = performance.now()
    var result = nn_run(nn, input)
    let nn_end = performance.now()

    // run post process.
    boxes = format_yolo_v2.run(result);
    boxes = nms.run(boxes)

    let end = performance.now()


    inferenceCtx.drawImage(video,
                  (video.videoWidth - video.videoHeight)/2, 0, video.videoHeight, video.videoHeight,
                  0, 0, 160, 160)

    canvasCtx.drawImage(video,
                  (video.videoWidth - video.videoHeight)/2, 0, video.videoHeight, video.videoHeight,
                  0, 0, canvas_size, canvas_size)

    let all_fps = (1000/(end - start)).toFixed(2)
    let network_fps = (1000/(nn_end - nn_start)).toFixed(2)
    canvasCtx.font = "30px Georgia";
    canvasCtx.fillText("FPS: " + all_fps, 10, 30);
    canvasCtx.font = "16px Georgia";
    canvasCtx.fillText("FPS (only network): " + network_fps, 10, 50);

    for (let box of boxes){
        canvasCtx.lineWidth = 4;
        canvasCtx.strokeStyle = "rgb(200, 75, 75)";
        canvasCtx.strokeRect(
            box[0] * canvas_size/160,
            box[1] * canvas_size/160,
            box[2] * canvas_size/160,
            box[3] * canvas_size/160
        )
    }

    // for debug
    //
    // counter++;
    // if( counter > 1000 ) {
    //     return;
    // }

    window.requestAnimationFrame(update)
}

const main = async () => {
    // Init Blueoil
    nn = nn_init()
    inputShape = nn_get_input_shape(nn)

    // Init cra
    navigator.mediaDevices = navigator.mediaDevices || ((navigator.mozGetUserMedia || navigator.webkitGetUserMedia) ? {
        getUserMedia: function(c) {
            return new Promise(function(y, n) {
                (navigator.mozGetUserMedia || navigator.webkitGetUserMedia).call(navigator, c, y, n)
            })
        }
    } : null)

    if (!navigator.mediaDevices) {
        alert('getUserMedia is not available in your browser')
        throw new Error('getUserMedia is not available in your browser')
    }

    const constraints = { audio: false, video: true }
    const stream = await navigator.mediaDevices.getUserMedia(constraints)

    video.srcObject = stream
    update()
}

const createResult = () => {
    const x = Math.random() * videoWidth / 2
    const y = Math.random() * videoHeight / 2
    const w = x + Math.random() * videoWidth
    const h = y + Math.random() * videoHeight
    return {
        coordinates: [x, y, w, h]
    }
}

const predict = () => {

}

document.addEventListener("DOMContentLoaded", function(event) {
    Module['onRuntimeInitialized'] = main
});
