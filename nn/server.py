"""Keep the model loaded on one GPU behind a small local JSON-over-HTTP API.

GET /info, POST /run, /effect, /attrib with the arguments of the matching nn.model
function plus "neuron" (e.g. "L12.N3456"). Requests are served one at a time.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import model as M
from .store import parse_neuron


def serve(model_name, host="127.0.0.1", port=8765, device="cuda"):
    model, tok = M.load(model_name, device)
    layers = M.decoder_layers(model)
    info = {"model": model_name, "layers": len(layers), "width": layers[0].mlp.down_proj.in_features}
    routes = {"/run": M.run, "/effect": M.effect, "/attrib": M.attrib}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, info) if self.path == "/info" else self.reply(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path not in routes:
                return self.reply(404, {"error": "unknown path"})
            try:
                args = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                layer, idx = parse_neuron(args.pop("neuron"))
                if not (0 <= layer < info["layers"] and 0 <= idx < info["width"]):
                    raise ValueError(f"neuron out of range: {info['layers']} layers x {info['width']} neurons")
                self.reply(200, routes[self.path](model, tok, layer, idx, **args))
            except Exception as e:  # report to the caller, keep serving
                self.reply(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    server = HTTPServer((host, port), Handler)
    print(f"serving {model_name} on http://{host}:{port}", flush=True)
    server.serve_forever()
