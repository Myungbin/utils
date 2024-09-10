from datetime import datetime


def get_current_time():
    return datetime.now()


def get_formatted_current_time(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)
