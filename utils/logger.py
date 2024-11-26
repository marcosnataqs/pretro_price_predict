from datetime import datetime, timezone
import json
import sys
import os
from io import BytesIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.lake_connector import LakeConnector


def upload_metrics_to_dl(metrics: json):
    try:
        # Save file into blob
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        metrics_file_name = f"logs/metrics_log_{timestamp}.json"
        metrics_json = json.dumps(metrics, indent=4).encode("utf-8")
        metrics_file = BytesIO(metrics_json)
        # Connnect to DL and Upload blob
        datalake_connector = LakeConnector("azure")
        blob = datalake_connector.connect(metrics_file_name)
        blob.upload_blob(metrics_file, overwrite=False)

    except Exception as e:
        print(f"Upload failed.{e}")
    else:
        print('Upload sucessful')
        # return True
