import logging
import time

# Get the program's start time
start_time = time.time()

class RelativeTimeFormatter(logging.Formatter):
    def formatTime(self, record):
        # Calculate the elapsed time since the start of the program
        elapsed_seconds = record.created - start_time
        # Convert to minutes and seconds
        minutes, seconds = divmod(elapsed_seconds, 60)
        # Return the formatted string
        return f"{int(minutes)}m{int(seconds):.0f}s"

    def format(self, record):
        record.relativeTime = self.formatTime(record)
        return super().format(record)

def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = RelativeTimeFormatter('%(relativeTime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
