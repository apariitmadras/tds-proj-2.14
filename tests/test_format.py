from utils import enforce_format, fabricate_fallback

def test_enforce_array_pad_trim():
    out = enforce_format(["a"], "JSON array", 3)
    assert out == ["a", "Unknown", "Unknown"]
    out = enforce_format(["a","b","c","d"], "JSON array", 2)
    assert out == ["a","b"]

def test_fabricate_object():
    qs = ["count?", "correlation?", "plot please"]
    out = fabricate_fallback(qs, "JSON object")
    assert isinstance(out, dict) and "answers" in out
    assert len(out["answers"]) == 3
