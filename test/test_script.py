import requests
import time
import tracemalloc

BASE_URL = "http://127.0.0.1:8000"  # Update with the actual FastAPI server URL

def test_file_recognize():
    file_path = "/path/to/test/image.jpg"  # Replace with a valid test image path
    response = requests.get(f"{BASE_URL}/fileRecognize", params={"filePath": file_path})
    print("fileRecognize response:", response.json())

def test_dir_recognize():
    dir_path = "/path/to/test/directory"  # Replace with a valid test directory path
    request_url = "http://example.com/api"  # Replace with a valid Java API endpoint
    task_id = "12345"
    model_name = "yolov8x.pt"
    response = requests.get(f"{BASE_URL}/dirRecognize", params={
        "dirPath": dir_path,
        "requestUrl": request_url,
        "taskId": task_id,
        "modelName": model_name
    })
    print("dirRecognize response:", response.json())

def test_voice_dir_recognize():
    dir_path = "/path/to/test/voice_directory"  # Replace with a valid test voice directory path
    request_url = "http://example.com/api"  # Replace with a valid Java API endpoint
    task_id = "67890"
    response = requests.get(f"{BASE_URL}/voiceDirRecognize", params={
        "dirPath": dir_path,
        "requestUrl": request_url,
        "taskId": task_id
    })
    print("voiceDirRecognize response:", response.json())

def test_memory_leaks():
    tracemalloc.start()
    for _ in range(10):  # Adjust the number of iterations as needed
        test_file_recognize()
        test_dir_recognize()
        test_voice_dir_recognize()
        time.sleep(1)  # Add a delay to simulate real-world usage
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    print("[ Top 10 memory usage ]")
    for stat in top_stats[:10]:
        print(stat)

if __name__ == "__main__":
    test_memory_leaks()
