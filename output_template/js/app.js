var em_module = require('./lib_js.js');
var network = require('./network.js');
var fs = require('fs');
var jpeg = require('jpeg-js');

// const sharp = require('sharp');

console.log(network);

var Jimp = require('jimp');


// main function, read from here
em_module['onRuntimeInitialized'] = main;
function main() {



    var jpegData = fs.readFileSync('image.jpg');
    var rawImageData = jpeg.decode(jpegData, true);

    network.uint8ArrayToFloat32Array;
    console.log(jpegData);
    console.log(rawImageData);



    var nn = network.init();
    var input_shape = network.get_input_shape(nn);
    console.log("input_shape", input_shape);

    console.log("aaa");
    var input_size = 1;
    for (var i = 0; i < input_shape.length; i++) {
        input_size *= input_shape[i];
    }

    Jimp.read('./image.jpg')
        .then(image => {
            console.log("image !!!!!!!")
            // console.log(image);
            image.resize(160, 160);
            console.log("aaa");
            console.log(image);
            console.log(image.bitmap.data);
            var file = "out.jpg"
            image.write(file);
            var data = image.getBufferAsync(Jimp.MIME_JPEG);
            console.log("bbb");
            console.log(data);
            data.then(i => {
                rawImageData = jpeg.decode(i, true);
                console.log(rawImageData);
                console.log(rawImageData.data.length);
            });
        });

    

    // var image_loader = sharp('./image.jpg')
    //     .resize(160)
    //     .toBuffer({ resolveWithObject: true });

    // image_loader.then(data => (
    //     console.log("data", data);
    // ))
    

    console.log("bbb");

    // console.log("ccc");
    // var input = new Float32Array(input_size);
    // // expects RGBRGBRGB...
    // for (var i = 0; i < input_size; i++) {
    //     var r = Math.random();
    //     input[i] = r;
    // }

    // console.log("ddd");
    // var result = network.run(nn, input);
    // console.log("eee");
    // var start = Date.now();
    // var trial = 100;
    // for (var i = 0; i < trial; i++) {
    //     var result = network.run(nn, input);
    // }
    // var end = Date.now();

    // console.log((end - start)/trial, "ms on average");
}

