
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
    def __init__(self, patience=6, monitor='loss', tol=1e-6):
        self.patience = patience
        self.cnt = 0
        self.prev_val = None
        self.tol = tol
        self.monitor = monitor
        
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
            print("Early stopping has occurred! The last state is: ", history_entry)
            model.should_stop = True