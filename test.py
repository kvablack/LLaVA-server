import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 4

# paths = glob.glob(f"target_paired_images_new_toykitchen7/*.jpg") * 100
paths = ["monkey.jpg"]

def f(_):
    for i in tqdm.tqdm(range(0, len(paths), BATCH_SIZE)):
        batch_paths = paths[i : i + BATCH_SIZE]

        jpeg_data = []
        queries = []
        answers = []
        for path in batch_paths:
            image = Image.open(path)

            # Compress the images using JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            jpeg_data.append(buffer.getvalue())

            queries.append([
                "What item is lying on the table?",
                "What color is the table?",
                "What is the robot arm doing?"
            ])
            answers.append(["a toy", "brown", "picking up a toy"])
            # texts.append(["What item is lying on the table?"])

        data = {"images": jpeg_data, "queries": queries, "answers": answers}
        data_bytes = pickle.dumps(data)

        # Send the JPEG data in an HTTP POST request to the server
        url = "http://34.70.249.53:8085"
        response = requests.post(url, data=data_bytes)

        # Print the response from the server
        response_data = pickle.loads(response.content)

        for output, score in zip(response_data["outputs"], response_data["precision"]):
            print(output)
            print(score)
            print("--")

with ThreadPoolExecutor(max_workers=10) as executor:
    for _ in executor.map(f, range(10)):
        pass