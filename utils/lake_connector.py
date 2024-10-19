import boto3
import os

from azure.storage.blob import BlobClient
from dotenv import load_dotenv
from typing import List, Any


class LakeConnector:
    def __init__(self, service_provider: str) -> None:
        self.available_services: List[str] = ["aws", "azure"]
        if service_provider.lower() not in self.available_services:
            raise ValueError(f"""Invalid service provider: {service_provider}. 
                             Available options are: {self.available_services}
                            """)
        self.service_provider: str = service_provider
        load_dotenv()

    def azure_connection(
        self, container_name: str, blob_name: str, connection_string: str
    ) -> BlobClient:
        blob = BlobClient.from_connection_string(
            conn_str=connection_string,
            container_name=container_name,
            blob_name=blob_name,
        )
        return blob

    def aws_connection(
        self, aws_access_key_id: str, aws_secret_access_key: str, aws_session_token: str
    ) -> boto3.client:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

        s3_client = session.client("s3")
        return s3_client

    def connect(self, file_path: str) -> Any:
        # to-do : adjust type hint
        connection: Any
        if self.service_provider.lower() == "azure":
            connection = self.azure_connection(
                container_name=os.environ["AZURE_CONTAINER_NAME"],
                blob_name=file_path,
                connection_string=os.environ["AZURE_CONNECTION_STRING_DL"],
            )
        elif self.service_provider.lower() == "aws":
            connection = self.aws_connection(
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            )

        return connection


if __name__ == "__main__":
    file_test = "raw/bitcoin/btc.csv"
    lc = LakeConnector("aws")
    connection = lc.connect(file_test)
    print(type(connection))
