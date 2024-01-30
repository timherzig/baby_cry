import torch

from src.data.data import get_dataloaders
from src.models.model import get_model


def test(config, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_out_file = "/".join(checkpoint.split("/")[:-2]) + "/test.txt"

    _, _, test_loader = get_dataloaders(config, device)

    Net = get_model(config, load_pretrained=False).to(device)
    check_point = torch.load(checkpoint)
    Net.load_state_dict(check_point["model_state_dict"])

    Net.eval()

    with torch.no_grad():
        for batch in test_loader:
            samples, _, _, labels = batch
            samples = samples.to(device)
            labels = labels.to(device)

            predictions = Net(samples)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            for prediction, label in zip(predictions, labels):
                with open(test_out_file, "a") as f:
                    f.write(f"{prediction}, {label}\n")
