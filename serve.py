from PIL import Image
from io import BytesIO
import http.server
import pickle
from functools import partial
import time
import traceback
from llava_server.llava import load_llava


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, inference_fn, *args, **kwargs):
        self.inference_fn = inference_fn
        super().__init__(*args, **kwargs)

    def do_POST(self):
        print(f"received POST request from {self.client_address}")

        try:
            t = time.time()
            content_length = int(self.headers["Content-Length"])
            data = self.rfile.read(content_length)
            print(f"read data in {time.time() - t:.2f}s")
            t = time.time()
            data = pickle.loads(data)
            print(f"deserialized data in {time.time() - t:.2f}s")

            t = time.time()
            images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
            print(f"decoded images in {time.time() - t:.2f}s")
            texts = data["texts"]

            t = time.time()
            response = self.inference_fn(images, texts)
            print(f"inference in {time.time() - t:.2f}s")

            t = time.time()
            response = pickle.dumps(response)
            print(f"serialized response in {time.time() - t:.2f}s")

            returncode = 200
        except Exception as e:
            response = traceback.format_exc()
            print(response)
            returncode = 500

        self.send_response(returncode)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(response)


HOST = "0.0.0.0"
PORT = 8085

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()

    inference_fn = load_llava(args.params_path)

    with http.server.HTTPServer(
        (HOST, PORT), partial(RequestHandler, inference_fn)
    ) as httpd:
        print(f"HTTP server listening on port {PORT}")
        httpd.serve_forever()
