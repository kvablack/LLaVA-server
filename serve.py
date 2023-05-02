from typing import Iterable
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from llava.conversation import Conversation, SeparatorStyle, conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPImageProcessor, StoppingCriteria

from PIL import Image

import os
from PIL import Image
from io import BytesIO

import http.server
import pickle
from functools import partial
import time
import traceback


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


PROMPT = "You are an assistant that is able to understand the visual content that the user provides. "
"You answer questions about the visual content in a short and concise manner.\n###Human: "


def load_model(params_path):
    # load model
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(params_path)
    model = AutoModelForCausalLM.from_pretrained(
        params_path, torch_dtype=torch.float16
    ).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_tower = model.model.vision_tower[0]
    vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    if mm_use_im_start_end:
        image_tokens = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            + DEFAULT_IM_END_TOKEN
        )
    else:
        image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    @torch.inference_mode()
    def inference_fn(
        images: Iterable[Image.Image], queries: Iterable[Iterable[str]]
    ) -> str:
        assert len(images) == len(queries)
        assert np.all(len(queries[0]) == len(q) for q in queries)

        queries = np.array(queries)  # (batch_size, num_queries_per_image)

        # preprocess images
        images = image_processor(images, return_tensors="pt")["pixel_values"]
        images = images.to("cuda", dtype=torch.float16)

        # first, get the activations for the image tokens
        initial_prompts = [PROMPT + image_tokens + "\n" for _ in range(len(images))]
        initial_input_ids = tokenizer(
            initial_prompts, return_tensors="pt"
        ).input_ids.cuda()
        initial_out = model(initial_input_ids, images=images, use_cache=True)
        initial_key_values = initial_out.past_key_values

        # broadcast the key values across the queries
        # becomes shape (batch_size * num_queries_per_image, ...)
        initial_key_values = [
            [
                x.unsqueeze(1)
                .expand(-1, queries.shape[1], -1, -1, -1)
                .reshape(-1, *x.shape[1:])
                for x in y
            ]
            for y in initial_key_values
        ]

        # flatten queries into one big batch
        flat_queries = queries.reshape(-1)  # (batch_size * num_queries_per_image)

        # prepare inputs for the queries
        prompts = [q + "###" for q in flat_queries]
        input_ids = tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.cuda()

        # stop upon seeing any of these tokens
        stop_tokens = torch.as_tensor(
            tokenizer.convert_tokens_to_ids(["▁###", "##", "#"]),
            dtype=torch.long,
            device="cuda",
        )

        # generation loop
        output_ids = []
        key_values = initial_key_values
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
        for i in range(50):
            out = model(input_ids=input_ids, use_cache=True, past_key_values=key_values)
            key_values = out.past_key_values
            next_tokens = torch.argmax(out.logits[:, -1], dim=-1)

            finished = finished | (next_tokens.unsqueeze(-1) == stop_tokens).any(dim=-1)

            if finished.all():
                break
            output_ids.append(next_tokens)
            input_ids = next_tokens.unsqueeze(-1)

        output_ids = torch.stack(output_ids, dim=-1)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # clean outputs
        outputs_clean = []
        for output in outputs:
            for pattern in ["###", "##", "#"]:
                if pattern in output:
                    output = output.split(pattern)[0]

            if "Assistant:" in output:
                output = output.split("Assistant:")[1]
            outputs_clean.append(output.strip())

        # reshape outputs back to (batch_size, num_queries_per_image)
        outputs_clean = np.array(outputs_clean).reshape(queries.shape)

        return outputs_clean.tolist()

    return inference_fn


class MyHandler(http.server.BaseHTTPRequestHandler):
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

    inference_fn = load_model(args.params_path)

    with http.server.HTTPServer(
        (HOST, PORT), partial(MyHandler, inference_fn)
    ) as httpd:
        print(f"HTTP server listening on port {PORT}")
        httpd.serve_forever()
