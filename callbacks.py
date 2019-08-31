from tools import save, load

import os
import pickle as pcl

class Callback:
    def __init__(self):
        pass

    def __call__(self):
        pass
    
    def restart(self):
        pass

class HistoryCallback(Callback):
    def __init__(self):
        pass

class ImmediateCallback(Callback):
    def __init__(self):
        pass

class EarlyStopping(HistoryCallback):
    def __init__(self, patience=6, monitor='loss', tol=1e-6, verbose=False):
        self.patience = patience
        self.cnt = 0
        self.prev_val = None
        self.tol = tol
        self.monitor = monitor
        self.verbose = verbose
        
    def restart(self):
        self.prev_val = None
        self.cnt = 0
    
    def __call__(self, history_entry, model):
        val = history_entry[self.monitor]
        has_improved = False
        if self.monitor.endswith('loss'):
            if self.prev_val is None or ((self.prev_val - val) > self.tol):
                has_improved = True
        else:
            if self.prev_val is None or ((val - self.prev_val) > self.tol):
                has_improved = True
        
        if has_improved:
            self.prev_val = val
            self.cnt = 0
        else:
            self.cnt += 1
        
        stop = False
        if self.cnt >= self.patience:            
            model.should_stop = True
            if self.verbose:
                print("Early stopping has occurred! The last state is: ", history_entry)


class ModelCheckpoint(HistoryCallback):
    def __init__(self, filepath, monitor='loss', save_best=True, verbose=False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best = save_best
        self.best_val = None
        self.best_filepath = None
        self.verbose = verbose
    
    def restart(self):
        self.best_val = None
        self.best_filepath = None
    
    def __call__(self, entry, model):
        val = entry[self.monitor]
        if not self.save_best:
            return self._write_to_path(entry, model)
        if self.monitor.endswith('loss'):
            if self.best_val is None or val < self.best_val:
                self.best_val = val
                return self._write_to_path(entry, model)
        else:
            if self.best_val is None or val > self.best_val:
                self.best_val = val
                return self._write_to_path(entry, model)
            
    
    def _write_to_path(self, entry, model):
        if self.save_best and self.best_filepath is not None:
            os.remove(self.best_filepath)
        self.best_filepath = self.filepath.format(**entry)
        save(model, self.best_filepath)
        if self.verbose:
            print(f"Found better model!, Written to file {self.best_filepath}")
    
    def load_best(self):
        if self.best_filepath:
            return load(self.best_filepath)