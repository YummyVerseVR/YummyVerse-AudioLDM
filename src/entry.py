import argparse
import json
import uvicorn
from api.app import App


SETTING_PATH = "./settings/meta.json"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--port",
    type=int,
    required=False,
    default=8001,
    help="port to run the server (default: 8001)",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="enable debug mode",
)
args = parser.parse_args()

with open(SETTING_PATH, "r") as f:
    settings = json.load(f)

app = App(settings, args.debug).get_app()

if __name__ == "__main__":
    uvicorn.run("entry:app", host="0.0.0.0", port=args.port, log_level="info")
