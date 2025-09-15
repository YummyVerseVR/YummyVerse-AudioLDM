import argparse
import uvicorn
from api.api import App

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    type=str,
    required=True,
    help="path to config .yaml file",
)
parser.add_argument(
    "-m",
    "--model-checkpoint",
    type=str,
    required=True,
    help="the checkpoint path for the model",
)
parser.add_argument(
    "-db",
    "--database-endpoint",
    type=str,
    required=False,
    default="http://localhost:8001",
    help="the endpoint for the database",
)
parser.add_argument(
    "-p",
    "--port",
    type=int,
    required=False,
    default=8003,
    help="port to run the server (default: 8003)",
)
parser.add_argument("--debug", action="store_true", help="enable debug mode")

args = parser.parse_args()
app = App(
    args.model_checkpoint, args.config, args.database_endpoint, args.debug
).get_app()

if __name__ == "__main__":
    uvicorn.run("entry:app", host="0.0.0.0", port=args.port, log_level="info")
