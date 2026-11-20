import logging

from discovery_agent.log_redaction import install_log_redaction
from discovery_agent.pii import mask_text

KEY = "AIzaSy" + "A" * 33


def _capture(fn):
    install_log_redaction()
    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    log = logging.getLogger("redaction-test")
    log.setLevel(logging.DEBUG)
    h = H()
    log.addHandler(h)
    try:
        fn(log)
    finally:
        log.removeHandler(h)
    return "\n".join(records)


def test_secret_in_message_and_args_is_masked():
    out = _capture(lambda log: log.warning("GET https://x.example/?key=%s failed", KEY))
    assert KEY not in out and "failed" in out


def test_secret_in_traceback_is_masked():
    def boom(log):
        try:
            raise RuntimeError(f"auth failed for API_KEY={KEY}")
        except RuntimeError:
            log.exception("planner failed")
    out = _capture(boom)
    assert KEY not in out and "planner failed" in out and "RuntimeError" in out


def test_plain_logs_are_unchanged():
    assert "run 123 ok" in _capture(lambda log: log.info("run %s ok", 123))


def test_user_facing_error_text_masks_secrets():
    assert KEY not in mask_text(f"Details: key={KEY}")
