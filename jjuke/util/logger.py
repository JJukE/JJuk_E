from datetime import datetime
from functools import reduce
from pathlib import Path

__all__ = ["CustomLogger", "timenow", "basic_config", "get_logger"]


class CustomLogger:
    def __init__(self, filename=None, filemode="a", use_color=True, lock=False):
        self.lock = lock
        self.empty = True

        if not lock:
            if filename is not None:
                self.empty = False
                filename = Path(filename)
                if filename.is_dir():
                    timestr = self._get_timestr().replace(" ", "_").replace(":", "-").replace(".", "-")
                    filename = filename / "log_{}.log".format(timestr)
                self.file = open(filename, filemode)
            else:
                self.empty = True

            self.use_color = use_color

    def _get_timestr(self):
        n = datetime.now()
        return "{:02d}.{:02d}.{:02d} {:02d}:{:02d}:{:02d}".format(
            n.year%100, n.month, n.day, n.hour, n.minute, n.second
        )

    def _write(self, msg, level):
        if self.lock:
            return

        timestr = self._get_timestr()
        out = "[{} {}] {}".format(timestr, level, msg)

        if self.use_color:
            if level == " INFO":
                # print("\033[32m" + out + "\033[0m")
                # print("\033[33m" + out + "\033[0m")
                # print("\033[34m" + out + "\033[0m")
                print("\033[96m" + out + "\033[0m")
                # print("\033[91m" + out + "\033[0m")
            elif level == " WARN":
                print("\033[35m" + out + "\033[0m")
            elif level == "ERROR":
                print("\033[31m" + out + "\033[0m")
            elif level == "FATAL":
                print("\033[43m\033[1m" + out + "\033[0m")
            else:
                print(out)
        else:
            print(out)

        if not self.empty:
            self.file.write(out + "\r\n")

    def debug(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "DEBUG")

    def info(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " INFO")

    def warn(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " WARN")

    def error(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "ERROR")

    def fatal(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "FATAL")

    def flush(self):
        if not self.lock and not self.empty:
            self.file.flush()


def timenow(braket=False):
    n = datetime.now()
    if braket:
        return "[{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}]".format(
            n.year%100, n.month, n.day, n.hour, n.minute, n.second
        )
    else:
        return "{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(
            n.year%100, n.month, n.day, n.hour, n.minute, n.second
        )


_logger = CustomLogger()


def basic_config(filename, lock=False):
    _logger.__init__(filename, lock=lock)


def get_logger():
    return _logger
