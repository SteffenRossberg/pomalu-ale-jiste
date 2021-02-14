import os


class Logger:

    @staticmethod
    def log(log_id, log_message):
        log_file_path = f'data/{log_id}.train.txt'
        mode = 'wt' if not os.path.exists(log_file_path) else 'at'
        with open(log_file_path, mode) as log_file:
            log_file.write(f"\n{log_message}")
