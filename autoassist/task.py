"""task.py -- step 2: decide binary / multiclass / regression from the target column."""
import pandas as pd


def detect_task(y, cfg):
    """Return (task, reason). Rules, in order:
       text / category / boolean target             -> classification
       integer target with <= max_int_classes values -> classification
       everything else (floats, many-valued ints)    -> regression
    Classification with exactly 2 classes is 'binary', otherwise 'multiclass'."""
    n_unique = int(y.nunique())
    if n_unique < 2:
        raise ValueError("The target has only %d distinct value; nothing to learn." % n_unique)

    if pd.api.types.is_bool_dtype(y):
        is_class, why = True, "target is boolean"
    elif pd.api.types.is_numeric_dtype(y):
        if pd.api.types.is_integer_dtype(y) and n_unique <= cfg.max_int_classes:
            is_class = True
            why = "integer target with only %d distinct values (<= %d)" % (n_unique, cfg.max_int_classes)
        elif pd.api.types.is_integer_dtype(y):
            is_class, why = False, "integer target with %d distinct values (> %d)" % (n_unique, cfg.max_int_classes)
        else:
            is_class, why = False, "target holds decimal numbers"
    else:
        is_class, why = True, "target is text"

    if not is_class:
        return "regression", why
    if n_unique > cfg.max_classes:
        raise ValueError("Target has %d distinct classes (max %d); is this really classification?"
                         % (n_unique, cfg.max_classes))
    return ("binary" if n_unique == 2 else "multiclass"), why + ", %d classes" % n_unique
