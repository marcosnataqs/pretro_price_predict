# LitServe => https://github.com/Lightning-AI/litserve

import os
import sys
import joblib
import litserve as ls
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model_training import PetroModel


class Petro_Predict_API(ls.LitAPI):
    def setup(self, device):
        # Set device to CPU
        device = torch.device("cpu")

        # Load the scaler
        self.input_scaler = joblib.load(
            os.path.join("ml_model_training", "input_scaler_petro.joblib")
        )

        self.output_scaler = joblib.load(
            os.path.join("ml_model_training", "output_scaler_petro.joblib")
        )

        # Load the LSTM model
        self.model = PetroModel(1, 7, 1, device)
        self.model.load_state_dict(
            torch.load(
                os.path.join("ml_model_training", "final_model.pth"),
                map_location=device,
                weights_only=True,
            )
        )
        self.model.eval()

        # Set the device
        self.device = device

        # Set up pipeline params
        # self.pipeline_params = {
        #     "num_lags": 7,
        #     "columns": ["pbr", "usd"],
        #     "num_features": 5,
        # }

    def decode_request(self, request):
        # Convert input to DataFrame
        input_data = pd.DataFrame(
            [request["input"]],
            columns=[
                "pbr_(t-7)",
                "pbr_(t-6)",
                "pbr_(t-5)",
                "pbr_(t-4)",
                "pbr_(t-3)",
                "pbr_(t-2)",
                "pbr_(t-1)",
            ],
        )
        X = self.input_scaler.transform(input_data)
        X = X.reshape((-1, 7, 1))
        X = torch.from_numpy(X.astype(np.float32))

        print(X)
        print(X.shape)

        # # Create PetroDataset
        # dataset = PetroDataset(input_data, self.pipeline_params)

        # Get the processed input
        # x, _ = input_data[0]

        # Add batch dimension and move to device
        # x = x.unsqueeze(0).to(self.device)

        return X

    def predict(self, x):
        with torch.no_grad():
            output = self.model(x)
        return output.cpu().numpy()

    def encode_response(self, output):
        # Inverse transform the output
        output_unscaled = self.output_scaler.inverse_transform(output)
        # output_unscaled = output
        return {"prediction": float(output_unscaled[0][0])}


if __name__ == "__main__":
    api = Petro_Predict_API()
    server = ls.LitServer(api)
    server.run(port=8000, generate_client_file=False)
