import os
import secrets

from flask import jsonify, request
from app import create_app
from app.config import Config

token = os.environ.get("CONTINGENCY_API_TOKEN", "")
if len(token) < 32:
    raise RuntimeError(
        "CONTINGENCY_API_TOKEN must contain at least 32 characters"
    )

errors = Config.validate()
if errors:
    raise RuntimeError("; ".join(errors))

app = create_app()
app.config["DEBUG"] = False


@app.before_request
def require_service_token():
    # Render can check service health without accessing project data.
    if request.path == "/health" and request.method == "GET":
        return None

    supplied = request.headers.get("Authorization", "")
    expected = f"Bearer {token}"

    if not secrets.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        return jsonify(error="Unauthorized"), 401
