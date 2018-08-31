import sys
import os

def lazy_init(obj, attr: str, attr_getter):
    if not hasattr(obj, attr) or getattr(obj, attr) is None:
        setattr(obj, attr, attr_getter())
    return getattr(obj, attr)

class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout