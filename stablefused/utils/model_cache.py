from typing import Any, Dict


class ModelCache:
    """
    A cache for diffusion models. This class should not be instantiated by the user.
    You should use the load_model function instead. It is a mapping from model_id to
    diffusion model. This allows us to avoid loading the same model components multiple
    times.
    """

    def __init__(self) -> None:
        self.cache = dict()

    def get(self, model_id: str, default: Any = None) -> Any:
        if model_id not in self.cache.keys():
            return default
        return self.cache[model_id]

    def set(self, model: Any) -> None:
        self.cache[model.model_id] = model


_model_cache = ModelCache()


load_model_from_cache = _model_cache.get
save_model_to_cache = _model_cache.set
