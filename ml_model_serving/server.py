# LitServe => https://github.com/Lightning-AI/litserve

import joblib
import numpy as np
import litserve as ls


class Petro_Predict_API(ls.LitAPI):
    def setup(self, device):
        self.model = joblib.load("pretro_price_predict_model.joblib")

    def decode_request(self, request):
        x = np.asarray(request["input"])
        x = np.expand_dims(x, 0)
        return x

    def predict(self, x):
        return self.model.predict(x)

    def encode_response(self, output):
        return {"class_idx": int(output)}


if __name__ == "__main__":
    api = Petro_Predict_API()
    server = ls.LitServer(api)
    server.run(port=8000)
