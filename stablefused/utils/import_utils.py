import importlib

from types import ModuleType
from typing import Optional


class LazyImporter:
    """
    Lazy importer for modules.
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.module = None

    def import_module(self, import_error_message: Optional[str] = None) -> ModuleType:
        if self.module is None:
            try:
                self.module = importlib.import_module(self.module_name)
            except ImportError:
                if import_error_message is None:
                    import_error_message = (
                        f"'{self.module_name}' is not installed. Please install it."
                    )
                raise ImportError(import_error_message)
        return self.module
