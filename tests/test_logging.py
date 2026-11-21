import logging

import pytest

from anomaly_detector.cli import main
from anomaly_detector.logging_config import ENV_VAR, LOGGER_NAME, resolve_level, setup_logging


@pytest.fixture(autouse=True)
def reset_logger():
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_resolve_level_names_and_ints():
    assert resolve_level("debug") == logging.DEBUG
    assert resolve_level(" Info ") == logging.INFO
    assert resolve_level(logging.ERROR) == logging.ERROR


def test_resolve_level_env_and_default(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert resolve_level() == logging.WARNING
    monkeypatch.setenv(ENV_VAR, "INFO")
    assert resolve_level() == logging.INFO


def test_resolve_level_invalid():
    with pytest.raises(ValueError, match="invalid log level"):
        resolve_level("LOUD")


def test_setup_logging_does_not_duplicate_handlers():
    setup_logging("INFO")
    logger = setup_logging("DEBUG")
    assert logger.level == logging.DEBUG
    assert len([h for h in logger.handlers if not isinstance(h, logging.NullHandler)]) == 1


def test_setup_logging_writes_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging("INFO", log_file=log_file)
    logging.getLogger("anomaly_detector.test").info("hello file")
    for h in logging.getLogger(LOGGER_NAME).handlers:
        h.flush()
    text = log_file.read_text()
    assert "hello file" in text and "INFO" in text


def test_cli_log_file_contains_pipeline_steps(tmp_path, csv_file, capsys):
    log_file = tmp_path / "run.log"
    code = main(["--log-level", "debug", "--log-file", str(log_file), "detect", str(csv_file)])
    assert code == 0
    text = log_file.read_text()
    for expected in ("loaded", "fitting", "pipeline finished"):
        assert expected in text


def test_cli_quiet_hides_info(csv_file, capsys):
    assert main(["-q", "detect", str(csv_file)]) == 0
    assert "INFO" not in capsys.readouterr().err


def test_cli_bad_env_level(monkeypatch, csv_file, capsys):
    monkeypatch.setenv(ENV_VAR, "LOUD")
    assert main(["detect", str(csv_file)]) == 2
    assert "invalid log level" in capsys.readouterr().err


def test_detector_bad_param_is_value_error():
    from anomaly_detector.detectors import get_detector

    with pytest.raises(ValueError, match="invalid parameters"):
        get_detector("zscore", window=5)
