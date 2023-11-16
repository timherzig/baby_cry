import os
import torch
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
import tf2onnx
import onnx


def convert_trillsson_hub_model(hub_path):
    input = layers.Input(shape=(None,))
    m = hub.KerasLayer(hub_path, trainable=False)
    embeddings = m(input)["embedding"]
    m = tf.keras.Model(inputs=m.input, outputs=embeddings)

    os.makedirs(f"tf_models/{hub_path.split('/')[-2]}", exist_ok=True)
    tf.saved_model.save(m, f"tf_models/{hub_path.split('/')[-2]}")


def convert_jdc_model(path):
    pass


def convert_local_model_to_torch(model_path):
    # m = torch.jit.load(model_path)

    # # Iterate through the model's layers and convert the weights and biases to PyTorch tensors
    # for param in m.parameters():
    #     param.requires_grad = False

    #     if len(param.shape) >= 2:
    #         torch.nn.init.xavier_uniform_(param)
    #     else:
    #         torch.nn.init.zeros_(param)

    # # Save the PyTorch model
    # torch.save(
    #     m.state_dict(), f"../models/torch_models/{model_path.split('/')[-2]}.pth"
    # )
    pass


if __name__ == "__main__":
    convert_trillsson_hub_model("https://tfhub.dev/google/trillsson1/1")
