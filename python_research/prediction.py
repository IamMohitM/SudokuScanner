from .torchserve_proto import inference_pb2 
from .torchserve_proto import inference_pb2_grpc
from tensorflow import make_tensor_proto

import torch
import grpc
import numpy as np
import requests

class DigitPredictor:
    def __init__(self, config) -> None:
        self.host = config["host"]
        self.model_name = config["model_name"]
        self.http_port = config["port"]
        self.grpc_port = config["grpc_port"]
        self.inference_url = self._get_pred_url()
        self.grpc_stub = self._get_stub()
        self.session = requests.Session()

    def _get_pred_url(self):
        return f"http://{self.host}:{self.http_port}/predictions/{self.model_name}"

    def _get_stub(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.grpc_port}")
        stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
        return stub

    def convert_to_bytes(self, img):
        return make_tensor_proto(img).tensor_content

    def make_grpc_prediction(self, img):

        image = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        image_as_bytes = self.convert_to_bytes(image)

        input_data = {"data": image_as_bytes, "shape": str(tuple(image.shape)).encode('utf-8')}
        

        prediction = self.grpc_stub.Predictions(inference_pb2.PredictionsRequest(model_name=self.model_name, input = input_data))

        return prediction

    def make_http_prediction(self, img):
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        image_as_bytes = self.convert_to_bytes

        input_data = {"data": image_as_bytes, "shape": str(tuple(img.shape)).encode('utf-8')}

        prediction = self.session.get(self.inference_url,data = input_data)
        return prediction.text
    