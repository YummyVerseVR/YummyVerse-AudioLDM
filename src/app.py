import requests
import os
import time
import asyncio
import yaml

from pydantic import BaseModel
from pylognet.client import LoggingClient, LogLevel

from fastapi import FastAPI, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

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
        self.__expire_time = config.get("expire", 300)
        self.__tasks: dict[str, dict] = {}
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
            "YummyAudioGenServer",
            self.__logger_endpoint,
            disable=not logging,
        )

        self.__model = AudioLDMController(
            checkpoint_path,
            yaml.load(open(config_path, "r"), Loader=yaml.FullLoader),
            self.__logger,
        )
        self.__model.set_savepath(self.__output_dir)

        self.__setup_routes()

    def __setup_routes(self):
        self.__app.add_event_handler("startup", self.startup_event)
        self.__router.post("/generate")(self.generate_audio)
        self.__router.get("/status/{task_id}")(self.get_status)
        self.__router.get("/queue")(self.queue_status)
        self.__router.get("/ping")(self.ping)

    def __generate(self, prompt: str, output_path: str):
        self.__model.generate_audio(
            output_path,
            prompt=prompt,
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

    def __background_generate(self, user_id: str, prompt: str):
        try:
            self.__tasks[user_id]["status"] = "processing"
            out_path = os.path.join(self.__output_dir, f"{user_id}.wav")
            self.__generate(prompt, out_path)
            self.__tasks[user_id]["status"] = "done"
            self.__tasks[user_id]["result"] = out_path
            self.__tasks[user_id]["timestamp"] = time.time()
            self.__save_to_db(user_id, out_path)
        except Exception as e:
            self.__tasks[user_id]["status"] = "error"
            self.__tasks[user_id]["error"] = str(e)

    async def __cleanup_expired_files(self):
        while True:
            now = time.time()
            expired = [
                tid
                for tid, t in self.__tasks.items()
                if "timestamp" in t and now - t["timestamp"] > self.__expire_time
            ]
            for tid in expired:
                result_path = self.__tasks[tid].get("result")
                if result_path and os.path.exists(result_path):
                    try:
                        os.remove(result_path)
                    except Exception:
                        pass
                del self.__tasks[tid]
            await asyncio.sleep(300)  # Check every 5 minutes

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # on_event("startup")
    async def startup_event(self):
        asyncio.create_task(self.__cleanup_expired_files())

    # /generate
    async def generate_audio(
        self,
        request: UserRequest,
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> JSONResponse:
        self.__tasks[request.user_id] = {"status": "pending", "timestamp": time.time()}
        background_tasks.add_task(
            self.__background_generate, request.user_id, request.prompt
        )

        self.__logger.log(
            f"Accepted audio generation task for user_id: {request.user_id}",
            LogLevel.INFO,
        )

        return JSONResponse(
            status_code=202,
            content={"message": "Task accepted", "task_id": request.user_id},
        )

    # /status/{task_id}
    async def get_status(self, user_id: str) -> JSONResponse:
        if user_id not in self.__tasks:
            return JSONResponse(status_code=404, content={"error": "Task not found"})
        return JSONResponse(
            content={"task_id": user_id, "status": self.__tasks[user_id]["status"]}
        )

    # /queue
    async def queue_status(self):
        return {
            "total": len(self.__tasks),
            "pending": [
                tid for tid, t in self.__tasks.items() if t["status"] == "pending"
            ],
            "processing": [
                tid for tid, t in self.__tasks.items() if t["status"] == "processing"
            ],
            "done": [tid for tid, t in self.__tasks.items() if t["status"] == "done"],
            "error": [tid for tid, t in self.__tasks.items() if t["status"] == "error"],
        }

    # /ping
    async def ping(self) -> JSONResponse:
        return JSONResponse(content={"message": "pong"})
