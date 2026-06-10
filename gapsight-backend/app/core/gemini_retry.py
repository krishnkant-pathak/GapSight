from __future__ import annotations

from google.genai.errors import ClientError, ServerError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def is_retryable_gemini_error(exc: BaseException) -> bool:
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        code = getattr(exc, "status_code", None)
        if code is None and exc.args:
            code = exc.args[0]
        return code == 429
    return False


GEMINI_RETRY = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=15, max=90),
    retry=retry_if_exception(is_retryable_gemini_error),
    reraise=True,
)
