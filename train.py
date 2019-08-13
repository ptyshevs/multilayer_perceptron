import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', default=argparse.SUPPRESS, help='Script to train model')
    parser.add_argument('-l', '--learning_rate', default=.05, type=float, help='Specify learning rate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose training process')
    
    args = parser.parse_args()
    
    
    
    