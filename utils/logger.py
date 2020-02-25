import os
import logging
import sys
import csv
from datetime import datetime

def setup_logger(name, save_dir, txt_file_name='log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "{}.txt".format(txt_file_name)))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_dict_into_csv(save_dir, file_name, output):
    try:
        # Append file_name with datetime
        #file_name = os.path.join(save_dir, file_name + "_{0:%Y-%m-%dT%H:%M:%S}".format(datetime.now()))
        file_name = os.path.join(save_dir, file_name)
        with open(file_name, "w") as file:
            writer = csv.writer(file)
            writer.writerow(output.keys())
            writer.writerows(zip(*output.values()))
    except IOError:
        print("Failed to save file {}".format(file_name))
