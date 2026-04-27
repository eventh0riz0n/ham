import os

# Remove external keys for a local-only demo. In production, set these in your shell instead.
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

from ham.memory_engine import MemoryEngine

engine = MemoryEngine()
try:
    engine.remember("Local-only HAM still works with deterministic fallback embeddings.", store="semantic")
    print(engine.recall("local fallback", top_k=1))
finally:
    engine.close()
