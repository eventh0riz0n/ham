from ham.memory_engine import MemoryEngine

engine = MemoryEngine()
try:
    ids = engine.remember(
        "HAM can store semantic memories and retrieve them later.",
        store="semantic",
        source="example",
        source_type="script",
        importance=0.8,
    )
    print("Remembered:", ids)

    for result in engine.recall("semantic memories", top_k=3):
        print(result["score"], result["store"], result["text"])
finally:
    engine.close()
