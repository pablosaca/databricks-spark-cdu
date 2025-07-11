import logging


def get_logger():
    """
    Configuración para logging
    """
    logger = logging.getLogger("SANTALUCÍA - PRUEBA TÉCNICA")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s | [Línea: %(lineno)d] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
