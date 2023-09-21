import torch
import tensorflow as tf
import tensorflow_hub as hub


def convert_hub_model_to_torch(hub_path):
    m = hub.KerasLayer(hub_path, trainable=False)

    tf.saved_model.save(m, f"tf_models/{hub_path.split('/')[-2]}")

    m = torch.jit.load(f"tf_models/{hub_path.split('/')[-2]}")

    # Iterate through the model's layers and convert the weights and biases to PyTorch tensors
    for param in m.parameters():
        param.requires_grad = False

        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

    # Save the PyTorch model
    torch.save(m.state_dict(), f"../models/torch_models/{hub_path.split('/')[-2]}.pth")


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
    convert_hub_model_to_torch("https://tfhub.dev/google/trillsson1/1")
