"""
@file logger.py
@author Benjamin Albert (balbert2@jhu.edu)
@author Chang Yan (cyan13@jhu.edu)
"""

from time import gmtime, strftime, localtime


class Logger:
    """
    Static utility class for printing time-stamped messages.
    The time displayed is in the local time zone.
    """
    @staticmethod
    def _get_time():
        return strftime("%Y-%m-%d %H:%M:%S", localtime())

    @staticmethod
    def log(msg):
        print("{} {}".format(strftime("%Y-%m-%d %H:%M:%S", localtime()), msg))
        file1 = open("log_re.txt","a")
        file1.write("{} {}\n".format(strftime("%Y-%m-%d %H:%M:%S", localtime()), msg))
        file1.close()