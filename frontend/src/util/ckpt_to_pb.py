import argparse
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ckpt to .pb")
    parser.add_argument("-c", "--trained_checkpoint", required=True, type=str, help="Choose trained checkpoint path")
    args = parser.parse_args()

    graph = tf.Graph()

    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(args.trained_checkpoint + '.meta')
        loader.restore(sess, args.trained_checkpoint)

        # Export checkpoint to SavedModel
        export_dir = args.trained_checkpoint.replace("ckpt", "pb")
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                            [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                            strip_default_attrs=True)
        builder.save()