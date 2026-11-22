"""The camera conversion: pose, lens, framing and the ground offset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cupid_camera import _demo


def test_camera_round_trip():
    _demo()


if __name__ == "__main__":
    test_camera_round_trip()
    print("test_camera: ok")
