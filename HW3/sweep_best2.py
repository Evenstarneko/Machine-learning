
import os

if __name__ == '__main__':
    for i in [1, 2, 3, 4, 5, 6]:
        command = "python main.py train --data-dir release-data --log-file best-logs-pool=" + str(i) + ".csv --model-save best-pool=" + str(i) + ".torch --model best --best-pool2 " + str(i) 
        os.system(command)
        