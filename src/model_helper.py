import tensorflow as tf
import collections



class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass


def create_train_model(
        model_creator,
        hparams):

    graph = tf.Graph()
    with graph.as_default(), tf.container("train"):
        model = model_creator(
                hparams,
                tf.contrib.learn.ModeKeys.TRAIN,
                )
        return TrainModel(
            graph=graph,
            model=model,
        )


