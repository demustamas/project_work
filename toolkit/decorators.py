import functools
from toolkit.logger import Logger


def log_args(func):
    """Logger decorator.

    It catches the inputs and outputs of a function and logs it to a logfile defined in toolkit.logger.

    Args:
        func (function): Function to log

    Raises:
        e: To catch the exception if happens

    Returns:
        function results: Results of the wrapped function
    """
    logger = Logger(func.__qualname__).get_logger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(
            f"{func.__name__} called with args: {signature}",
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.exception(
                f"Exception raised in {func.__name__}. exception: {str(e)}"
            )
            raise e

    return wrapper
