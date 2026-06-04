import logging
import os
from contextlib import contextmanager


BROKEN_LOCAL_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
}
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_TRANSFORMER_LOAD_LOGGERS = (
    "transformers.core_model_loading",
    "transformers.integrations.peft",
    "transformers.integrations.tensor_parallel",
    "transformers.modeling_utils",
    "transformers.utils.loading_report",
)


def clear_broken_proxy_env() -> None:
    cleared = []
    for name in PROXY_ENV_VARS:
        value = (os.getenv(name) or "").strip()
        if value.lower() in BROKEN_LOCAL_PROXY_VALUES:
            os.environ.pop(name, None)
            cleared.append(name)
    if cleared:
        logging.getLogger(__name__).warning(
            "Cleared broken proxy variables: %s",
            ", ".join(sorted(cleared)),
        )


def is_closed_huggingface_client_error(exc: Exception) -> bool:
    return "client has been closed" in str(exc).lower()


def reset_huggingface_session_if_closed() -> None:
    try:
        from huggingface_hub.utils import close_session, get_session
    except Exception:
        return

    try:
        session = get_session()
    except Exception:
        return

    if getattr(session, "is_closed", False):
        logging.getLogger(__name__).warning("Resetting closed Hugging Face HTTP client.")
        close_session()


def recover_closed_huggingface_session(exc: Exception) -> bool:
    if not is_closed_huggingface_client_error(exc):
        return False

    try:
        from huggingface_hub.utils import close_session
    except Exception:
        return False

    logging.getLogger(__name__).warning("Retrying after closed Hugging Face HTTP client.")
    close_session()
    return True


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
