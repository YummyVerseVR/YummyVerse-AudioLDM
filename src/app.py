import requests
import os
import asyncio
import yaml

from pydantic import BaseModel
from pylognet.client import LoggingClient, LogLevel

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor

from controller import AudioLDMController


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(
        self,
        config: dict,
        debug: bool = False,
        logging: bool = False,
    ):
        self.__app = FastAPI()
        self.__router = APIRouter()
        config_path = config.get("config", "./config/kaggle_custom.yaml")
        checkpoint_path = config.get("checkpoint", "./data/checkpoints/trained.ckpt")
        self.__output_dir = config.get("output", "./output")
        self.__debug = debug
        self.__endpoints = config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        os.makedirs(self.__output_dir, exist_ok=True)

        self.__logger_endpoint = self.__endpoints.get(
            "logger", "http://logger.local:9000"
        )

        self.__logger = LoggingClient(
            "YummyAudioServer",
            self.__logger_endpoint,
            disable=not logging,
        )

        self.__queue = asyncio.Queue()
        self.__executor = ThreadPoolExecutor()
        self.__model = AudioLDMController(
            checkpoint_path,
            yaml.load(open(config_path, "r"), Loader=yaml.FullLoader),
            self.__logger,
        )
        self.__model.set_savepath(self.__output_dir)

        self.__setup_routes()
        self.__logger.log(
            "Audio Generation Server initialized successfully",
            LogLevel.INFO,
        )

    def __setup_routes(self):
        self.__router.add_event_handler("startup", self.__activate)

        self.__router.add_api_route(
            "/generate",
            self.generate,
            methods=["POST"],
        )
        self.__router.add_api_route(
            "/ping",
            self.ping,
            methods=["GET"],
        )

    def __save_to_db(self, user_id: str, audio_path: str):
        if self.__debug:
            self.__logger.log(
                "Database server connection skipped in debug mode",
                LogLevel.DEBUG,
            )
            return

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"user_id": user_id}
                self.__logger.log(
                    f"Saving generated audio to DB for user_id: {user_id}",
                    LogLevel.INFO,
                )
                response = requests.post(
                    f"{self.__control_endpoint}/save/audio", files=files, data=data
                )
                os.remove(audio_path)
                response.raise_for_status()
        except Exception as e:
            self.__logger.log(
                f"Failed to save generated audio to DB for user_id: {user_id}: {e}",
                LogLevel.ERROR,
            )

    def __generate(self, user_id: str, prompt: str):
        try:
            out_path = os.path.join(self.__output_dir, f"{user_id}.wav")
            self.__model.generate_audio(
                out_path,
                prompt=prompt,
            )
            self.__logger.log(
                f"Completed audio generation for prompt: {prompt}, saved to {out_path}",
                LogLevel.INFO,
            )
            self.__save_to_db(user_id, out_path)
        except Exception as e:
            self.__logger.log(
                f"Error during audio generation for prompt: {prompt}: {e}",
                LogLevel.ERROR,
            )

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    async def __worker(self):
        while True:
            item = await self.__queue.get()
            self.__executor.submit(self.__generate, item.user_id, item.prompt)
            self.__queue.task_done()
            await asyncio.sleep(5)

    def __activate(self) -> None:
        asyncio.create_task(self.__worker())

    # /generate
    async def generate(
        self,
        request: UserRequest,
    ) -> JSONResponse:
        self.__logger.log(
            f"Accepted audio generation task for prompt: {request.prompt}",
            LogLevel.INFO,
        )
        await self.__queue.put(request)

        return JSONResponse(
            status_code=202,
            content={"message": "Task enqueued", "user_id": request.user_id},
        )

    # /ping
    async def ping(self) -> JSONResponse:
        return JSONResponse(content={"message": "pong"})
