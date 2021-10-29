
import os

if __name__ == '__main__':
    for i in [10, 30, 50, 70, 90, 110, 130]:
        command = "python main.py train --data-dir release-data --log-file best-logs-blf=" + str(i) + ".csv --model-save best-blf=" + str(i) + ".torch --model best --best-linear-features " + str(i) 
        os.system(command)
        