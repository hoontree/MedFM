import sys
import os
from unittest.mock import MagicMock

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from distillers import create_distiller, UnifiedDistiller


def test_create_distiller():
    print("Testing create_distiller factory...")

    class MockConfig:
        def __init__(self, method_name):
            self.method = {
                "name": method_name,
                "alpha": 1.0,
                "beta": 0.0,
                "gamma": 0.0,
                "gamma_attn": 0.0,
                "gamma_align": 0.0,
                "use_dice": True,
                "use_ce": True,
                "layer_mapping": {},
                "layer_channels": {},
            }
            # Add get method to simulate DictConfig.get
            self.method_dict = self.method
            self.method = MagicMock()
            self.method.name = method_name
            self.method.get.side_effect = lambda k, default=None: self.method_dict.get(
                k, default
            )

            self.data = MagicMock()
            self.data.num_classes = 1

            self.teacher = MagicMock()
            self.student = MagicMock()

    # Names that should all resolve to UnifiedDistiller
    names = ["logit", "feature", "adaptive_layer", "hybrid", "unified"]

    for name in names:
        cfg = MockConfig(name)
        try:
            distiller = create_distiller(cfg)
            print(f"create_distiller('{name}') -> {type(distiller).__name__}")
            assert isinstance(distiller, UnifiedDistiller)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"create_distiller('{name}') failed: {e}")
            return False

    print("Verification Success!")
    return True


if __name__ == "__main__":
    if not test_create_distiller():
        sys.exit(1)
