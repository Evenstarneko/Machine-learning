import numpy as np

def main():
    file = np.load('./Data/0/cropbox.npz')
    a = file['a']
    print('Hello')

if __name__ == '__main__':
    main()