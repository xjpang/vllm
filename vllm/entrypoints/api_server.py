"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import json
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    # lora
    lora_id = request_dict.pop("lora_id", None)
    lora_path = request_dict.pop("lora_path", None)
    if lora_id is None or lora_path is None:
        lora_request = None
    else:
        lora_request = LoRARequest(lora_name=lora_id, lora_int_id=0, lora_local_path=lora_path)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    # jimpang add
    prompt_token_ids = None
    if prompt and len(prompt) > 0:
        first_element = prompt[0]
        if isinstance(first_element, int):
            prompt_token_ids = prompt
            prompt = None

    results_generator = engine.generate(
        prompt=prompt, sampling_params=sampling_params, request_id=request_id, prompt_token_ids=prompt_token_ids,
        lora_request=lora_request)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            output_tokens = [output.token_ids for output in request_output.outputs]
            ret = {"text": text_outputs, "output_token_ids": output_tokens}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]
    output_tokens = [output.token_ids for output in final_output.outputs]
    ret = {"text": text_outputs, "output_token_ids": output_tokens}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)

    app.root_path = args.root_path

    logger.info("Available routes are:")
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)
        logger.info("Route: %s, Methods: %s", route.path, methods)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
