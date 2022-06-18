from sudoku import SudokuScanner
import json
import logging

log_format = "%(levelname)s [%(lineno)s] [%(message)s]"
logging.basicConfig(format=log_format, level=logging.DEBUG)

if __name__ == "__main__":
    with open("configs/config.json", 'r') as f:
        config = json.load(f)
    sudoku_scanner = SudokuScanner(config)
    sudoku_scanner.run()
    pass