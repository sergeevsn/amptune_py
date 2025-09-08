import segyio
import numpy as np

def read_segy(file_path):
    """
    Read a segy file and return the trace data 
    as numpy array with shape (n_traces, n_samples) 
    and the sample interval in seconds
    """
    try:
        with segyio.open(file_path, ignore_geometry=True) as segy_file:
            return segy_file.trace.raw[:], segy_file.bin[segyio.BinField.Interval]*1e-06
    except Exception as e:
        print(f"Error reading segy file {file_path}: {e}")
        return None, None

def write_segy(target_path, reference_path, data, sample_interval):
    """
    Write a segy file from a numpy array with shape (n_traces, n_samples)
    and the sample interval in seconds. 
    The reference path is used to get trace headers.
    The target path is the path to the new segy file.
    The data is the numpy array with shape (n_traces, n_samples).
    The sample interval is the sample interval in seconds.
    """
    try:
        with segyio.open(reference_path, ignore_geometry=True) as segy_file:
            spec = segyio.spec()
            spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
            spec.format = segyio.SegySampleFormat.IBM_FLOAT_4_BYTE
            spec.samples = np.arange(data.shape[1])*sample_interval*1000 # in milliseconds
            spec.tracecount = data.shape[0]     # number of traces   
        
            with segyio.create(target_path, spec=spec) as target_file:
                target_file.bin[segyio.BinField.Interval] = int(sample_interval*1e06)
                target_file.bin[segyio.BinField.Samples] = data.shape[1]
                for i in range(data.shape[0]):
                    target_file.trace[i] = data[i]
                    target_file.header[i] = segy_file.header[i]
    except Exception as e:
        print(f"Error writing segy file {target_path}: {e}")