import logging
from contextlib import contextmanager


_TRANSFORMER_LOAD_LOGGERS = (
    "transformers.core_model_loading",
    "transformers.integrations.peft",
    "transformers.integrations.tensor_parallel",
    "transformers.modeling_utils",
    "transformers.utils.loading_report",
)


@contextmanager
def quiet_transformer_loading():
    original_levels = []
    try:
        for name in _TRANSFORMER_LOAD_LOGGERS:
            logger = logging.getLogger(name)
            original_levels.append((logger, logger.level))
            if logger.level < logging.ERROR:
                logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level in original_levels:
            logger.setLevel(level)
