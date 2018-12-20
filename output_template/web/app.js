// Ready for init blueoil
let nn
let inputShape

// Init app
const video = document.getElementById('video')
const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')
// const canvasCopy = document.getElementById("canvascopy");
// const copyContext = canvasCopy.getContext('2d');

const videoWidth = video.offsetWidth
const videoHeight = video.offsetHeight

canvas.width  = 160;
canvas.height = 160;

canvas.setAttribute("width", 160);
canvas.setAttribute("height", 160);


var counter = 0;

const update = () => {
	// Get image data
	const imageData = ctx.getImageData(0, 0, 160, 160)

    format_yolo_v2 = new FormatYoloV2([160, 160], 1, [1, 2, 3, 4, 5], 5, "NCHW");

    console.log(format_yolo_v2.num_cell);

    console.log("AAAAAA");
    console.log(imageData);

	// このデータがCameraキャプチャデータなので、これを使って推論してください

	// 以下ワッキーが書いたサンプル
	var input_size = 1

	// Make dummy data
	for (var i = 0; i < inputShape.length; i++) {
		input_size *= inputShape[i]
	}
	var input = new Float32Array(input_size)

    j = 0;
    for (var i = 0; i < imageData.data.length; i++) {
        input[j] = imageData.data[i] / 255.0;
        if ((i % 4) != 0 ){
            j++;
        }
    }
    var max = d3.max(input);

    if (max > 0.7) {

        console.log("run")
        var result = nn_run(nn, input)

        console.log(result)

        format_yolo_v2.run(result);

        return
    }


	// var result = nn_run(nn, input)
	// console.log(result)


	// console.log(input_size)
	// const input
	// var input_shape = network.get_input_shape(nn);

	// Cameraのキャプチャをキャンバスに描画
    ctx.drawImage(video, 0, 0, 160, 160)

    counter++;

    if( counter > 10 ) {
        return;
    }
    window.requestAnimationFrame(update)
}

const main = async () => {
	// Init Blueoil
	nn = init()
	inputShape = nn_get_input_shape(nn)
    console.log("inputshape", inputShape)

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

	// ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

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










// var em_module = require('./lib_js.js');
// var network = require('./network.js');
// var fs = require('fs');
// var jpeg = require('jpeg-js');

// var post_process = require('./post_process.js');
// // const sharp = require('sharp');

// console.log(network);

// var Jimp = require('jimp');


// // main function, read from here
// em_module['onRuntimeInitialized'] = main;
// function main() {
//     format_yolo_v2 = new post_process.FormatYoloV2([160, 160], ["a", "b"], [1, 2, 3], "NCHW");

//     console.log(format_yolo_v2.num_cell);


//     var jpegData = fs.readFileSync('image.jpg');
//     var rawImageData = jpeg.decode(jpegData, true);

//     network.uint8ArrayToFloat32Array;
//     console.log(jpegData);
//     console.log(rawImageData);



//     var nn = network.init();
//     var input_shape = network.get_input_shape(nn);
//     console.log("input_shape", input_shape);

//     console.log("aaa");
//     var input_size = 1;
//     for (var i = 0; i < input_shape.length; i++) {
//         input_size *= input_shape[i];
//     }

//     var input = new Float32Array(input_size);
//     var runner = Jimp.read('./image.jpg')
//         .then(image => {
//             console.log("image !!!!!!!")
//             // console.log(image);
//             image.resize(160, 160);
//             // console.log("aaa");
//             // console.log(image);
//             // console.log(image.bitmap.data);
//             var file = "out.jpg"
//             image.write(file);
//             var data = image.getBufferAsync(Jimp.MIME_JPEG);

//             return data;
//         })
//         .then(img => {
//             rawImageData = jpeg.decode(img, true);
//             console.log(rawImageData);
//             console.log(rawImageData.data.length);

//             j = 0;
//             for (var i = 0; i < rawImageData.data.length; i++) {
//                 input[j] = rawImageData.data[i] / 255.0;
//                 if ((i % 4) != 0 ){
//                     j++;
//                 }
//             }
//             var result = network.run(nn, input);
//             // console.log(result);

//             return result
//         });

//     runner.then(result => {
//         var aa = format_yolo_v2.run(result);
//         console.log("ccccccc")
//         console.log(aa);
//     })

//     // expects RGBRGBRGB...


//     // var image_loader = sharp('./image.jpg')
//     //     .resize(160)
//     //     .toBuffer({ resolveWithObject: true });

//     // image_loader.then(data => (
//     //     console.log("data", data);
//     // ))

//     console.log("bbb");

//     console.log("ccc");


//     // console.log("ddd");
//     var result = network.run(nn, input);
//     // console.log("eee");
//     // var start = Date.now();
//     // var trial = 100;
//     // for (var i = 0; i < trial; i++) {
//     //     var result = network.run(nn, input);
//     // }
//     // var end = Date.now();

//     // console.log((end - start)/trial, "ms on average");
// }

