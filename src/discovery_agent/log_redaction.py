"""Mask API keys and tokens in every log record, including tracebacks.

``logger.exception`` and third-party loggers (httpx logs full request URLs,
which can contain ``?key=...``) write to the console unmasked.  A log-record
factory sees every record in the process, so the masking cannot be bypassed by
logging from a module that forgot to call a helper.  Only secrets are masked
here, not general PII, so log lines stay useful for debugging.
"""

from __future__ import annotations

import logging
import traceback

from .pii import _mask_secrets

_installed = False


def install_log_redaction() -> None:
    """Idempotent.  Safe to call from several modules."""
    global _installed
    if _installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        try:
            record.msg = _mask_secrets(record.getMessage())
            record.args = None
            if record.exc_info and record.exc_info[0] is not None:
                record.exc_text = _mask_secrets("".join(traceback.format_exception(*record.exc_info)).rstrip())
                record.exc_info = None
            if record.stack_info:
                record.stack_info = _mask_secrets(record.stack_info)
        except Exception:  # never let redaction break logging
            record.msg = "[log message withheld: redaction failed]"
            record.args = None
            record.exc_info = None
            record.exc_text = None
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
