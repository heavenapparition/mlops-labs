import bentoml
import numpy as np
import onnxruntime as rt
from pathlib import Path
from typing import Annotated
import onnx
import os
import mlflow
import time
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import gradio as gr
import tempfile

os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://localhost:9000"

mlflow.set_tracking_uri('http://localhost:9909')
experiment_name = "bento_service"
mlflow.set_experiment(experiment_name)
mlflow.autolog()

onnx_model = onnx.load('raw_xgb.onnx')
bentoml.onnx.save_model('raw_xgb', onnx_model)



@bentoml.service(
    name="service",
    resources={"cpu": "8"},
    traffic={"timeout": 10},
)
class Svc(bentoml.Service):
    def __init__(self):
        self.model = bentoml.onnx.load_model('raw_xgb')

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        X_array = df.values
        X_dict = {input_name: X_array.astype(np.float32)}
        return label_name, X_dict

    def preprocess_for_prediction(self, df: pd.DataFrame) -> np.ndarray:
        data = deepcopy(df)
        categorical_features = data.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature].astype(str))

        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return data

    @bentoml.api
    def predict(self, input_file: str) -> str:
        # Create a unique output filename
        temp_dir = tempfile.mkdtemp()
        output_file = f'predictions_{int(time.time())}.csv'
        output_path = os.path.join(temp_dir, output_file)
        
        with mlflow.start_run():
            df = pd.read_csv(input_file)
            df_preprocessed = self.preprocess_for_prediction(df)
            label_name, X_dict = self.preprocess(df_preprocessed)
            
            start_time = time.time()
            predictions = self.model.run([label_name], X_dict)
            execution_time = time.time() - start_time
            mlflow.log_metric("execution_time_seconds", execution_time)
            
            np.savetxt(output_path, predictions[0], delimiter=',')
            return output_path

    def create_ui(self):
        def predict_file(file):
            return self.predict(file.name)

        demo = gr.Interface(
            fn=predict_file,
            inputs=gr.File(label="Input CSV file"),
            outputs=gr.File(label="Predictions CSV"),
            title="XGBoost Prediction Service",
            description="Upload a CSV file to get predictions"
        )
        
        return demo

if __name__ == "__main__":
    svc = Svc()
    demo = svc.create_ui()
    demo.launch(server_port=3000)