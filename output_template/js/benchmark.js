var em_module = require('./lib_js.js');

function convertToUint8Array(typedArray) {
  var numBytes = typedArray.length * typedArray.BYTES_PER_ELEMENT;
  var ptr = em_module._malloc(numBytes);
  var heapBytes = new Uint8Array(em_module.HEAPU8.buffer, ptr, numBytes);
  heapBytes.set(new Uint8Array(typedArray.buffer));
  return heapBytes;
}

function uint8ArrayToFloat32Array(a) {
    var nDataBytes = a.length * a.BYTES_PER_ELEMENT;
    var dataHeap = new Float32Array(a.buffer, a.byteOffset, a.length/4);
    dataHeap.set(a.buffer);
    return dataHeap;
}

function uint8ArrayToInt32Array(a) {
    var nDataBytes = a.length * a.BYTES_PER_ELEMENT;
    var dataHeap = new Int32Array(a.buffer, a.byteOffset, a.length/4);
    dataHeap.set(a.buffer);
    return dataHeap;
}

function _freeArray(heapBytes) {
  em_module._free(heapBytes.byteOffset);
}

function init() {
    var network_create = em_module.cwrap("network_create", "number", []);
    var network_init = em_module.cwrap("network_init", "bool", ["number"]);
    var nn = network_create();
    network_init(nn);
    return nn;
}

function  nn_get_input_shape(nn) {
    var network_get_input_rank = em_module.cwrap("network_get_input_rank", "number", ["number"]);
    var network_get_input_shape = em_module.cwrap("network_get_input_shape", "", ["number", "number"]);
    var input_rank = network_get_input_rank(nn);
    var input_shape = new Int32Array(input_rank);
    var input_shape_ = convertToUint8Array(input_shape);
    network_get_input_shape(nn, input_shape_.byteOffset);
    return uint8ArrayToInt32Array(input_shape_);
}

function  nn_get_output_shape(nn) {
    var network_get_output_rank = em_module.cwrap("network_get_output_rank", "number", ["number"]);
    var network_get_output_shape = em_module.cwrap("network_get_output_shape", "", ["number", "number"]);
    var output_rank = network_get_output_rank(nn);
    var output_shape = new Int32Array(output_rank);
    var output_shape_ = convertToUint8Array(output_shape);
    network_get_output_shape(nn, output_shape_.byteOffset);
    return uint8ArrayToInt32Array(output_shape_);
}

function nn_run(nn, input) {
    var output_shape = nn_get_output_shape(nn);
    var output_size = 1;

    for (var i = 0; i < output_shape.length; i++) {
        output_size *= output_shape[i];
    }

    console.log("d1");
    var output = new Float32Array(output_size);
    var input_ = convertToUint8Array(input);
    var output_ = convertToUint8Array(output);

    console.log("d2");
    var network_run = em_module.cwrap("network_run", "", ["number", "number", "number"]);
    var start = Date.now();
    network_run(nn, input_.byteOffset, output_.byteOffset);

    var end = Date.now();
    console.log("elapsed:", (end - start));

    var r = uint8ArrayToFloat32Array(output_);

    _freeArray(input_);
    _freeArray(output_);
    return r;
}

// main function, read from here
em_module['onRuntimeInitialized'] = onRuntimeInitialized;
function onRuntimeInitialized() {

    var nn = init();
    console.log("aaa");
    var input_size = 1;
    var input_shape = nn_get_input_shape(nn);
    console.log("bbb");
    for (var i = 0; i < input_shape.length; i++) {
        input_size *= input_shape[i];
    }

    console.log("ccc");
    var input = new Float32Array(input_size);
    // expects RGBRGBRGB...
    for (var i = 0; i < input_size; i++) {
        var r = Math.random();
        input[i] = r;
    }

    console.log("ddd");
    var result = nn_run(nn, input);
    console.log("eee");
    var start = Date.now();
    var trial = 100;
    for (var i = 0; i < trial; i++) {
        var result = nn_run(nn, input);
    }
    var end = Date.now();

    console.log((end - start)/trial, "ms on average");
}
