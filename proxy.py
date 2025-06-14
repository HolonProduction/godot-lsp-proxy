from contextlib import suppress
from enum import StrEnum
from functools import wraps
import json
import asyncio
from asyncio import StreamReader, StreamWriter
from typing import Any, Dict, List, Optional, cast
import os
from urllib.parse import urlparse

PROXY_HOST = "localhost"
PROXY_PORT = 6007

GODOT_HOST = "localhost"
GODOT_PORT = 6005

CLIENT_PATH = "/home/user"
SERVER_PATH = "/wsl.localhost/Ubuntu/home/user"

# Might transform unwanted stuff in markup content. But links in there will probably work.
TRANFORM_MARKUP = True

META_MODEL_LOCATIONS = ["meta_model.json", "godot_custom_endpoints.json"]


class TransformDirection(StrEnum):
    client_to_server = "client_to_server"
    server_to_client = "server_to_client"


def DebugCall(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


class Transformer:
    @DebugCall
    def transform(self, container, direction: TransformDirection):
        return container


class MapTransformer(Transformer):
    def __init__(
        self,
        key_tansformer: Optional[Transformer],
        value_transformer: Optional[Transformer],
    ):
        self.key_tansformer = key_tansformer or Transformer()
        self.value_transformer = value_transformer or Transformer()

    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not isinstance(container, dict):
            return container
        result = {}

        for k in container.keys():
            transformed_key = self.key_tansformer.transform(k, direction)
            transformed_value = self.value_transformer.transform(
                container[k], direction
            )
            result[transformed_key] = transformed_value

        return result


class ListTransformer(Transformer):
    def __init__(self, value_transformer: Transformer) -> None:
        self.value_transformer = value_transformer

    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not isinstance(container, list):
            return container
        return list(
            map(lambda x: self.value_transformer.transform(x, direction), container)
        )


class MarkupTransformer(Transformer):
    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not TRANFORM_MARKUP:
            return container
        if not isinstance(container, dict):
            return container
        value: str = container["value"]

        if direction == TransformDirection.client_to_server:
            from_path = CLIENT_PATH
            to_path = SERVER_PATH
        elif direction == TransformDirection.server_to_client:
            from_path = SERVER_PATH
            to_path = CLIENT_PATH

        container["value"] = value.replace(from_path, to_path)
        return container


class KeyedTransformer(Transformer):
    def __init__(self, transformer_map: Dict[str, Transformer]) -> None:
        self.transformer_map = transformer_map

    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not isinstance(container, dict):
            return container
        for k in container:
            if k in self.transformer_map:
                container[k] = self.transformer_map[k].transform(
                    container[k], direction
                )
        return container


class TupleTransformer(Transformer):
    def __init__(self, transformer_map: Dict[int, Transformer]) -> None:
        self.transformer_map = transformer_map

    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not isinstance(container, list):
            return container
        for k in self.transformer_map:
            if k < len(container):
                container[k] = self.transformer_map[k].transform(
                    container[k], direction
                )
        return container


class UriTransformer(Transformer):
    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not isinstance(container, str):
            return container

        parsed = urlparse(container)
        if parsed.scheme != "file":
            return container

        path = os.path.abspath(os.path.join(parsed.netloc, parsed.path))

        # Assuming CLIENT_PATH and SERVER_PATH are URI encoded.
        if direction == TransformDirection.client_to_server:
            from_path = CLIENT_PATH
            to_path = SERVER_PATH
        elif direction == TransformDirection.server_to_client:
            from_path = SERVER_PATH
            to_path = CLIENT_PATH

        rel = os.path.relpath(path, from_path)
        if rel.startswith(".."):
            return container

        transformed_path = os.path.join(to_path, rel)
        uri = "file://" + transformed_path
        return uri


class PathTransformer(Transformer):
    @DebugCall
    def transform(self, container, direction: TransformDirection):
        if not isinstance(container, str):
            return container

        if direction == TransformDirection.client_to_server:
            from_path = CLIENT_PATH
            to_path = SERVER_PATH
        elif direction == TransformDirection.server_to_client:
            from_path = SERVER_PATH
            to_path = CLIENT_PATH

        rel = os.path.relpath(container, from_path)
        if rel.startswith(".."):
            return container

        transformed_path = os.path.join(to_path, rel)
        return transformed_path


class MultiTransformer(Transformer):
    def __init__(self, transformers: List[Transformer]) -> None:
        self.transformers = transformers

    @DebugCall
    def transform(self, container, direction: TransformDirection):
        c = container
        for t in self.transformers:
            c = t.transform(c, direction)
        return c


class LazyTransformer(Transformer):
    deffered: Optional[Transformer]

    def __init__(self, name: str) -> None:
        self.name = name
        self.deffered = None

    def transform(self, container, direction: TransformDirection):
        if self.deffered is None:
            return container
        return self.deffered.transform(container, direction)


class LSPTransformer:
    id_map: Dict[Any, str]
    request_map: Dict[str, Transformer]
    response_map: Dict[str, Transformer]

    def __init__(self):
        self.id_map = {}
        self.request_map = {}
        self.response_map = {}

        enumerations = {}
        type_aliases = {}
        structures = {}

        def visit_type(deffinition: dict) -> Optional[Transformer]:
            match deffinition["kind"]:
                case "base":
                    if deffinition["name"] in ["URI", "DocumentUri"]:
                        return UriTransformer()
                    else:
                        return None
                case "array":
                    value_transform = visit_type(deffinition["element"])
                    if value_transform:
                        return ListTransformer(value_transform)
                    return None
                case "map":
                    key_transformer = visit_type(deffinition["key"])
                    value_transformer = visit_type(deffinition["value"])
                    if key_transformer or value_transformer:
                        return MapTransformer(key_transformer, value_transformer)
                    return None
                case "and":
                    transformers = list(
                        filter(
                            lambda x: x is not None,
                            map(lambda x: visit_type(x), deffinition["items"]),
                        )
                    )
                    if len(transformers):
                        return MultiTransformer(cast(List[Transformer], transformers))
                    return None
                case "or":
                    transformers = list(
                        filter(
                            lambda x: x is not None,
                            map(lambda x: visit_type(x), deffinition["items"]),
                        )
                    )
                    if len(transformers):
                        return MultiTransformer(cast(List[Transformer], transformers))
                    return None
                case "tuple":
                    transformers = list(
                        map(lambda x: visit_type(x), deffinition["items"])
                    )
                    keyed = {}
                    for i in range(len(transformers)):
                        if transformers[i] is not None:
                            keyed[i] = transformers[i]

                    if len(keyed):
                        return TupleTransformer(keyed)
                    return None
                case "reference":
                    name = deffinition["name"]
                    if name in enumerations:
                        return None
                    if name in type_aliases:
                        if name == "Path":
                            return PathTransformer()

                        entry = type_aliases[name]
                        if len(entry) == 1:
                            entry.append(LazyTransformer(name))
                            res = visit_type(type_aliases[name][0])
                            entry[1].deffered = res
                            return res
                        return entry[1]
                    if name in structures:
                        if name == "MarkupContent":
                            return MarkupTransformer()

                        entry = structures[name]
                        if len(entry) == 1:
                            entry.append(LazyTransformer(name))
                            struct = structures[name][0]
                            extends = (
                                cast(
                                    List[Transformer],
                                    list(
                                        filter(
                                            lambda x: x is not None,
                                            map(
                                                lambda x: visit_type(x),
                                                struct["extends"],
                                            ),
                                        )
                                    ),
                                )
                                if "extends" in struct
                                else []
                            )
                            mixins = (
                                cast(
                                    List[Transformer],
                                    list(
                                        filter(
                                            lambda x: x is not None,
                                            map(
                                                lambda x: visit_type(x),
                                                struct["mixins"],
                                            ),
                                        )
                                    ),
                                )
                                if "mixins" in struct
                                else []
                            )
                            keyed = {}
                            for prop in struct["properties"]:
                                transformer = visit_type(prop["type"])
                                if transformer is not None:
                                    keyed[prop["name"]] = transformer
                            transformers = extends + mixins
                            if len(keyed):
                                transformers = [KeyedTransformer(keyed)] + transformers
                            if len(transformers):
                                tr = MultiTransformer(transformers)
                                entry[1].deffered = tr
                                return tr
                            return None
                        return entry[1]
            return None

        for location in META_MODEL_LOCATIONS:
            with open(location, "r") as file:
                model: dict = json.load(file)

            for enum in model["enumerations"]:
                enumerations[enum["name"]] = True

            for t in model["typeAliases"]:
                type_aliases[t["name"]] = [t["type"]]

            for t in model["structures"]:
                structures[t["name"]] = [t]

            for request in model["requests"] + model["notifications"]:
                if "params" not in request:
                    continue
                method = request["method"]
                params = request["params"]

                if isinstance(params, dict):
                    self.request_map[method] = visit_type(params) or Transformer()
                elif isinstance(params, dict):
                    transformers = list(map(lambda x: visit_type(x), params))
                    keyed = {}
                    for i in range(len(transformers)):
                        if transformers[i] is not None:
                            keyed[i] = transformers[i]

                    if len(keyed):
                        self.request_map[method] = TupleTransformer(keyed)
                    else:
                        self.request_map[method] = Transformer()

            for response in model["requests"]:
                method = response["method"]
                result = response["result"]

                self.response_map[method] = visit_type(result) or Transformer()

    def transform(self, message_string: str, direction: TransformDirection) -> str:
        try:
            message: dict = json.loads(message_string)
            if "method" in message:
                method = message["method"]
                if "id" in message:
                    self.id_map[message["id"]] = method
                if "params" not in message:
                    print("Did not contain params nothing to transform.")
                    return message_string
                if method not in self.request_map:
                    print(f"Couldn't transform request, for unkown method {method}")
                    return message_string
                message["params"] = self.request_map[method].transform(
                    message["params"], direction
                )
            elif "result" in message:
                if not "id" in message or message["id"] not in self.id_map:
                    print("Couldn't transform response, due to missing or unknown id")
                    return message_string
                method = self.id_map.pop(message["id"])
                if method not in self.response_map:
                    print(f"Couldn't transform response, for unkown method {method}")
                    return message_string
                message["result"] = self.response_map[method].transform(
                    message["result"], direction
                )
            else:
                print(
                    "Could not map object, which was no request, response or notification"
                )

            return json.dumps(message)
        except Exception as e:
            print(e)
            return message_string


async def recv_lsp_message(reader: StreamReader) -> Optional[str]:
    """Read an LSP message from the TCP stream in a blocking manner."""

    content_length = None

    # Read header
    res = await reader.readuntil("\r\n\r\n".encode("ascii"))
    header = res.decode("ascii")
    for field in header.split("\r\n"):
        parts = field.split(":")
        if len(parts) != 2:
            continue
        if parts[0] == "Content-Length":
            content_length = int(parts[1].strip())
            break

    if content_length == None:
        raise Exception()

    content = await reader.readexactly(content_length)

    if not content:
        return None

    return content.decode("utf-8")


def make_lsp_package(message: str) -> bytes:
    message_encoded = message.encode("utf-8")
    header = f"Content-Length: {len(message_encoded)}\r\n\r\n".encode("ascii")

    return header + message_encoded


async def tcp_proxy(
    origin: StreamReader,
    target: StreamWriter,
    transformer: LSPTransformer,
    direction: TransformDirection,
):
    while True:
        message = await recv_lsp_message(origin)
        if not message:
            return

        message = transformer.transform(message, direction)

        target.write(make_lsp_package(message))


async def handle_client(client_reader: StreamReader, client_writer: StreamWriter):
    print("LSP client connection taken")
    print("Connecting to Godot Language Server")

    server_reader, server_writer = await asyncio.open_connection(GODOT_HOST, GODOT_PORT)
    print("Connected to Godot Language Server")

    transformer = LSPTransformer()

    request_task = asyncio.create_task(
        tcp_proxy(
            client_reader,
            server_writer,
            transformer,
            TransformDirection.client_to_server,
        )
    )
    response_task = asyncio.create_task(
        tcp_proxy(
            server_reader,
            client_writer,
            transformer,
            TransformDirection.server_to_client,
        )
    )

    done, pending = await asyncio.wait(
        [request_task, response_task], return_when=asyncio.FIRST_EXCEPTION
    )
    for p in pending:
        p.cancel()
        with suppress(asyncio.CancelledError):
            await p

    print("Disconnected")


async def main():
    server = await asyncio.start_server(handle_client, PROXY_HOST, PROXY_PORT)
    async with server:
        print("Waiting for LSP client connections.")
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped server")
