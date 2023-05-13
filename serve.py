from PIL import Image
from io import BytesIO
import http.server
import pickle
from functools import partial
import traceback
from llava_server.llava import load_llava
from llava_server.bertscore import load_bertscore
import numpy as np


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, inference_fn, bertscore_fn, *args, **kwargs):
        self.inference_fn = inference_fn
        self.bertscore_fn = bertscore_fn
        super().__init__(*args, **kwargs)

    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            data = self.rfile.read(content_length)

            # expects a dict with "images", "queries", and optionally "answers"
            # images: (batch_size,) of JPEG bytes
            # queries: (batch_size, num_queries_per_image) of strings
            # answers: (batch_size, num_queries_per_image) of strings
            data = pickle.loads(data)

            images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
            queries = data["queries"]

            print(f"Got {len(images)} images, {len(queries[0])} queries per image")

            outputs = self.inference_fn(images, queries)

            response = {"outputs": outputs}

            if "answers" in data:
                print(f"Running bertscore...")
                output_shape = np.array(outputs).shape
                (
                    response["precision"],
                    response["recall"],
                    response["f1"],
                ) = self.bertscore_fn(
                    np.array(outputs).reshape(-1).tolist(),
                    np.array(data["answers"]).reshape(-1).tolist(),
                )

                for key in ["precision", "recall", "f1"]:
                    response[key] = response[key].reshape(output_shape).tolist()

            # returns: a dict with "outputs" and optionally "scores"
            # outputs: (batch_size, num_queries_per_image) of strings
            # precision: (batch_size, num_queries_per_image) of floats
            # recall: (batch_size, num_queries_per_image) of floats
            # f1: (batch_size, num_queries_per_image) of floats
            response = pickle.dumps(response)

            returncode = 200
        except Exception as e:
            response = traceback.format_exc()
            print(response)
            response = response.encode("utf-8")
            returncode = 500

        self.send_response(returncode)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(response)


HOST = "127.0.0.1"
PORT = 8085

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()

    inference_fn = load_llava(args.params_path)
    bertscore_fn = load_bertscore()

    with http.server.HTTPServer(
        (HOST, PORT), partial(RequestHandler, inference_fn, bertscore_fn)
    ) as httpd:
        print(f"HTTP server listening on {HOST}:{PORT}")
        httpd.serve_forever()
