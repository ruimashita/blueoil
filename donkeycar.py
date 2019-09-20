import tensorflow as tf
# from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

# Function to export Keras model to Protocol Buffer format
# Inputs:
#       path_to_h5: Path to Keras h5 model
#       export_path: Path to store Protocol Buffer model

def export_h5_to_pb(path_to_h5, export_path):

    # Set the learning phase to Test since the model is already trained.
    tf.keras.backend.set_learning_phase(0)

    # Load the Keras model
    keras_model = tf.keras.models.load_model(path_to_h5)


    # import ipdb; ipdb.set_trace()
    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={"img_in": keras_model.input},
                                      outputs={
                                          "scores": keras_model.output[0],
                                          "scores2": keras_model.output[1],
                                      })

    # Build the Protocol Buffer SavedModel at 'export_path'
    # builder = saved_model_builder.SavedModelBuilder(export_path)

    with tf.keras.backend.get_session() as sess:
        # Save the meta graph and the variables

        output_node_names = ["n_outputs0/BiasAdd", "n_outputs1/BiasAdd"]


        # import ipdb; ipdb.set_trace()
        minimal_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            output_node_names,
        )

        pb_name = "minimal_graph_with_shape.pb"
        tf.train.write_graph(minimal_graph_def, export_path, pb_name, as_text=False)

        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input': keras_model.input},
            outputs={t.name:t for t in keras_model.output})

    #     builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
    #                                      signature_def_map={"predict": signature})

    # builder.save()




def main():
    path_to_h5 = "/home/wakisaka/mycar/models/mypilot2.h5"
    export_path = "export"

    export_h5_to_pb(path_to_h5, export_path)


if __name__ == "__main__":
    main()
