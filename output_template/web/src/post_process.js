const softmax = (arr) => {
    let max = d3.max(arr)

    console.log("max", max);
    var exp = new Float32Array(arr.length);
    var out = new Float32Array(arr.length);

    var j = 0;
    for (var i of arr){
        exp[j] = Math.exp(i - max);
        j++;
    }

    let sum = d3.sum(exp);
    console.log("sum", sum);
    var j = 0;
    for (var i of exp){
        out[j] = i / sum
        j++;
    }
    console.log("out", out);
    return out
}

const sigmoid = (arr) => {
    var out = new Float32Array(arr.length);

    var j = 0;
    for (var i of arr){
        out[j] = 1 / (1 + Math.exp(-i))
        j++;
    }

    console.log("sig out", out);
    return out
}





class FormatYoloV2 {
    constructor(image_size, num_classes, anchors, boxes_per_cell, data_format) {
        this.image_size = image_size;
        this.num_classes = num_classes;
        this.anchors = anchors;
        this.boxes_per_cell = boxes_per_cell
        this.data_format = data_format;
    }
    get num_cell(){
        return [this.image_size[0] / 32, this.image_size[1] / 32];
    }



    run(array){
        var num_cell_y = this.num_cell[0];
        var num_cell_x = this.num_cell[1];

        if (array.length != num_cell_y * num_cell_x * this.boxes_per_cell * (this.num_classes + 5)){
            console.log(array.length, num_cell_y * num_cell_x * this.boxes_per_cell *  (this.num_classes + 5))
            console.error("error");
            return
        }
        var batch_size = 1;

        // softmax(array);

        for (var i = 0; i < num_cell_y; i++){
        for (var j = 0; j < num_cell_x; j++){
            for (var k = 0; k < this.boxes_per_cell; k++){

                var index = i * j * k * (this.num_classes + 5);
                var next_index = index + this.num_classes + 5;

                var predictions = array.slice(index, next_index);
                console.log("predictions", predictions)

                var predict_classes = predictions.slice(0, this.num_classes);
                predict_classes = softmax(predict_classes)

                //console.log("predict_classes", predict_classes)
                var predict_confidence = array.slice(this.num_classes, this.num_classes + 1)
                predict_confidence = sigmoid(predict_confidence)

                var x = sigmoid(array.slice(this.num_classes + 1, this.num_classes + 2))
                var y = sigmoid(array.slice(this.num_classes + 2, this.num_classes + 3))
                var w = array.slice(this.num_classes + 3, this.num_classes + 4)
                var h = array.slice(this.num_classes + 4, this.num_classes + 5)

                // console.log("predict_confidence", predict_confidence)
        }
        }

        }

        console.log("format end")
    }

}


