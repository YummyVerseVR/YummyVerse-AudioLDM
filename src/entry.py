import argparse
import uvicorn
from api.api import App

if __name__ == "__main__":
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
        "-e",
        "--database-endpoint",
        type=str,
        required=True,
        help="the endpoint for the vector database",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=9000,
        help="the port to run the server on",
    )

    args = parser.parse_args()
    app = App(args.model_checkpoint, args.config, args.database_endpoint)
    uvicorn.run(app.get_app(), host="0.0.0.0", port=args.port, log_level="info")
