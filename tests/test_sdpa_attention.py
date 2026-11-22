"""The sdpa replacement for the sparse attention paths, against plain softmax."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "vendor" / "Cupid" / "cupid" / "modules" / "sparse" / "attention"))

from sdpa_groups import _demo


def test_sdpa_matches_naive_softmax():
    _demo()


if __name__ == "__main__":
    test_sdpa_matches_naive_softmax()
    print("test_sdpa_attention: ok")
