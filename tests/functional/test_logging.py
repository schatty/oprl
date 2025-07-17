from logging import Logger

from oprl.logging import (
    LoggerProtocol,
    create_stdout_logger,
    make_text_logger_func,
)


def test_create_stdout_logger() -> None:
    logger = create_stdout_logger()
    assert isinstance(logger, Logger)


def test_create_text_logger_func() -> None:
    func = make_text_logger_func("test_algo", "test_env")
    logger = func(0)
    assert isinstance(logger, LoggerProtocol)

