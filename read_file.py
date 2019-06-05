import pandas as pd
import numpy as np
import pdb


def read_file(filename, label='class', sep=None):
    
    if filename.split('.')[-1] == 'gz':
        compression = 'gzip'
    else:
        compression = None

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                engine='python')

    # let's check if I have a 'target' column:
    set_of_col_names = set(input_data.columns)
    set_of_labels = {'Label', 'Class', 'class', 'target'}
    labels_in_dataset = set_of_col_names.intersection(set_of_labels)
    err_msg = "Columns in dataset in file '%s': \n%s\n" % (filename, set_of_col_names)
    if len(labels_in_dataset) == 0:
        err_msg += "Data does not contain a target column (has to be one of %s)" % (set_of_labels)
        raise RuntimeError(err_msg)
    elif len(labels_in_dataset) > 1:
        err_msg += "Data contains more than 1 target column (has to be exactly one of %s)" % (set_of_labels)
        raise RuntimeError(err_msg)

    input_data.rename(columns={'Label': 'class','Class':'class', 'target':'class'},
                      inplace=True)

    feature_names = np.array([x for x in input_data.columns.values if x != label])

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, feature_names
