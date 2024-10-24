# LitServe => https://github.com/Lightning-AI/litserve

import os
import sys
import joblib
import litserve as ls
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model_training import PetroModel, PetroDataset


class Petro_Predict_API(ls.LitAPI):
    def setup(self, device):
        # Load the scaler
        self.scaler = joblib.load(
            os.path.join("ml_model_training", "scaler_petro.joblib")
        )

        # Load the LSTM model
        self.model = PetroModel(1, 7, 1, device)
        self.model.load_state_dict(
            torch.load(
                os.path.join("ml_model_training", "final_model.pth"),
                map_location=device,
            )
        )
        self.model.eval()

        # Set the device
        self.device = device

        # Set up pipeline params
        self.pipeline_params = {
            "num_lags": 7,
            "columns": ["pbr", "usd"],
            "num_features": 5,
        }

    def decode_request(self, request):
        # Convert input to DataFrame
        input_data = pd.DataFrame(
            [request["input"]],
            columns=["pbr", "brent", "wti", "production", "usd"],
        )

        # Create PetroDataset
        dataset = PetroDataset(input_data, self.pipeline_params)

        # Get the processed input
        x, _ = dataset[0]

        # Add batch dimension and move to device
        x = x.unsqueeze(0).to(self.device)

        return x

    def predict(self, x):
        with torch.no_grad():
            output = self.model(x)
        return output.cpu().numpy()

    def encode_response(self, output):
        # Inverse transform the output
        output_unscaled = self.scaler.inverse_transform(output)
        return {"prediction": float(output_unscaled[0][0])}


if __name__ == "__main__":
    api = Petro_Predict_API()
    server = ls.LitServer(api)
    server.run(port=8000, generate_client_file=False)