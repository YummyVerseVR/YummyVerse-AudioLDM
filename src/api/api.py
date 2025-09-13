from fastapi import FastAPI, APIRouter, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import requests
import os
import time
import asyncio
import yaml

from .controller import AudioLDMController


class App:
    OUTPUT_DIR = "./outputs"
    EXPIRE_SECONDS = 3600

    def __init__(
        self,
        checkpoint_path: str = "./latest.ckpt",
        config_path: str = "./config.yaml",
        db_endpoint: str = "http://localhost:8000",
    ):
        self.__app = FastAPI()
        self.__router = APIRouter()
        self.__model = AudioLDMController(
            checkpoint_path,
            yaml.load(open(config_path, "r"), Loader=yaml.FullLoader),
        )
        self.__tasks: dict[str, dict] = {}
        self.__db_endpoint = db_endpoint
        os.makedirs(App.OUTPUT_DIR, exist_ok=True)

        self.__setup_routes()

    def __setup_routes(self):
        self.__app.add_event_handler("startup", self.startup_event)
        self.__router.post("/generate/")(self.generate_audio)
        self.__router.get("/status/{task_id}")(self.get_status)
        self.__router.get("/download/{task_id}")(self.download_result)
        self.__router.get("/queue")(self.queue_status)

    def __generate(self, prompt: str, output_path: str):
        self.__model.generate_audio(
            output_path,
            prompt=prompt,
        )

    def __save_to_db(self, user_id: str, audio_path: str):
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"user_id": user_id}
                response = requests.post(
                    f"{self.__db_endpoint}/save/audio", files=files, data=data
                )
                response.raise_for_status()
        except Exception as e:
            print(f"Failed to save to DB: {e}")

    def __background_generate(self, user_id: str, prompt: str):
        try:
            self.__tasks[user_id]["status"] = "processing"
            out_path = os.path.join(App.OUTPUT_DIR, f"{user_id}.wav")
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
                if "timestamp" in t and now - t["timestamp"] > App.EXPIRE_SECONDS
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

    # @app.on_event("startup")
    async def startup_event(self):
        asyncio.create_task(self.__cleanup_expired_files())

    # @app.post("/generate/")
    async def generate_audio(
        self,
        user_id: str = Form(...),
        prompt: str = Form(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ):
        self.__tasks[user_id] = {"status": "pending", "timestamp": time.time()}
        background_tasks.add_task(self.__background_generate, user_id, prompt)
        return {"user_id": user_id}

    # @app.get("/status/{task_id}")
    async def get_status(self, user_id: str):
        if user_id not in self.__tasks:
            return JSONResponse(status_code=404, content={"error": "Task not found"})
        return self.__tasks[user_id]

    # @app.get("/download/{task_id}")
    async def download_result(self, task_id: str):
        task = self.__tasks.get(task_id)
        if not task or task.get("status") != "done":
            return JSONResponse(status_code=404, content={"error": "Result not ready"})
        return FileResponse(
            task["result"],
            media_type="audio/wav",
            filename=os.path.basename(task["result"]),
        )

    # @app.get("/queue")
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
