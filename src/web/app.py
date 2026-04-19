"""
Web server — Flask API + reference UI.

Run: python -m web.app
Open: http://localhost:5000
"""

import os
import logging
from flask import Flask, request, jsonify, send_from_directory

from engine import MemoryEngine, EngineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

app = Flask(__name__, static_folder="static")

# Initialize engine
config = EngineConfig(
    db_url=os.environ.get("DATABASE_URL", "postgresql://localhost:5432/memory_engine"),
    goals=[
        "memory engine", "consolidation", "AI infrastructure",
        "startup", "deep tech", "neuroscience", "defence",
    ],
)

engine = MemoryEngine(config)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/remember", methods=["POST"])
def remember():
    data = request.json
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "empty content"}), 400

    source = data.get("source", "direct")
    context = data.get("context", "")

    mem = engine.remember(content, source=source, context=context)
    return jsonify(mem)


@app.route("/api/recall", methods=["POST"])
def recall():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400

    top_k = data.get("top_k", 10)
    results = engine.recall(query, top_k=top_k)
    return jsonify(results)


@app.route("/api/consolidate", methods=["POST"])
def consolidate():
    stats = engine.consolidate()
    return jsonify(stats)


@app.route("/api/forget", methods=["POST"])
def forget():
    data = request.json
    memory_id = data.get("memory_id")
    below_temp = data.get("below_temp")
    result = engine.forget(memory_id=memory_id, below_temp=below_temp)
    return jsonify(result)


@app.route("/api/memories")
def memories():
    limit = request.args.get("limit", 300, type=int)
    return jsonify(engine.get_all(limit=limit))


@app.route("/api/memory/<mid>")
def memory(mid):
    mem = engine.get_memory(mid)
    if not mem:
        return jsonify({"error": "not found"}), 404
    return jsonify(mem)


@app.route("/api/stats")
def stats():
    return jsonify(engine.stats())


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


if __name__ == "__main__":
    print("\n  Memory Engine v0.1 — Web Server")
    print(f"  DB:  {config.db_url}")
    print(f"  Compression: deterministic (token importance)")
    print(f"\n  Open http://localhost:5000\n")
    app.run(debug=True, port=5001)
