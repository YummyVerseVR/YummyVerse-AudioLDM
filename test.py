import shutil
import os


def test(path):
    for dir in os.listdir(path):
        for file in os.listdir(f"{path}/{dir}"):
            shutil.move(f"{path}/{dir}/{file}", f"{path}")
            shutil.rmtree(f"{path}/{dir}")


if __name__ == "__main__":
    test("test")
