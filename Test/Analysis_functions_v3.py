# %%
import os
import gc
import sys
import shlex
import cairosvg
import subprocess
import numpy as np
import qcodes as qc
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import savgol_filter, find_peaks
from qcodes.parameters import Parameter
from IPython.display import clear_output
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset import (
    Measurement, initialise_or_create_database_at,
    load_or_create_experiment, plot_dataset)


def open_plottr(db_path, os_type='auto'):
    # Determine the OS type if not specified
    if os_type == 'auto':
        os_type = 'Mac' if sys.platform == 'darwin' else 'Windows' if sys.platform == 'win32' else 'Unknown'

    # Construct the command based on OS type
    if os_type == 'Windows':
        script_path = rf"C:\Users\admin\SynologyDrive\00 Users\Albert\10_Data\03_Data analysis\run_plottr.bat"
        command = [script_path, db_path]
    elif os_type == 'Mac':
        script_path = r"/Users/albert-mac/Library/CloudStorage/SynologyDrive-KeLab/00 Users/Albert/10_Data/03_Data analysis/run_plottr.sh"
        command = f"/bin/zsh {shlex.quote(script_path)} {shlex.quote(db_path)}"
    else:
        raise ValueError(f"Unsupported OS type: {os_type}")

    # Execute the command
    try:
        if os_type == 'Mac':
            subprocess.Popen(command, shell=True)
        else:
            subprocess.Popen(command)
        print(f"Plottr opened successfully for {os_type}")
    except subprocess.SubprocessError as e:
        print(f"Error opening Plottr: {e}")
    
    clear_output(wait=True)

def _display_time(run_time):
    total_seconds = int(run_time.total_seconds())

    if total_seconds < 60:
        display_time = f"{total_seconds} sec"
    elif total_seconds < 3600:
        minutes, seconds = divmod(total_seconds, 60)
        display_time = f"{minutes} min {seconds} sec" if seconds else f"{minutes} min"
    else:
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        display_time = f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    return display_time 

def get_dataset_info(dataset):
    run_id = dataset.captured_run_id
    exp_name = dataset.exp_name
    sample_name = dataset.sample_name
    description = dataset.description
    # 假設 run_timestamp() 和 completed_timestamp() 返回的是字串格式的時間戳，需要將它們轉換為 datetime 對象
    from datetime import datetime
    start_time = datetime.strptime(dataset.run_timestamp(), '%Y-%m-%d %H:%M:%S')
    completed_time = datetime.strptime(dataset.completed_timestamp(), '%Y-%m-%d %H:%M:%S')
    # 計算運行時間
    run_time = completed_time - start_time
    display_time = _display_time(run_time)
    return run_id, exp_name, sample_name, description, start_time, completed_time, run_time, display_time



def auto_select_unit(value, original_unit):
    """
    Automatically select appropriate unit for values.
    Supports V, A, T, Hz, Ω, and Ω/A.
    """
    abs_value = abs(value)
    unit_prefixes = {
        'V': ['nV', 'µV', 'mV', 'V', 'kV', 'MV'],
        'A': ['nA', 'µA', 'mA', 'A', 'kA', 'MA'],
        'T': ['nT', 'µT', 'mT', 'T'],
        'Hz': ['mHz', 'Hz', 'kHz', 'MHz', 'GHz'],
        'Ω': ['mΩ', 'Ω', 'kΩ', 'MΩ', 'GΩ'],
        'Ω/A': ['Ω/A', 'kΩ/A', 'MΩ/A', 'GΩ/A']
    }
    
    if original_unit not in unit_prefixes:
        return value, original_unit

    prefixes = unit_prefixes[original_unit]
    powers = range(-3 * (len(prefixes) // 2), 3 * (len(prefixes) - len(prefixes) // 2), 3)
    
    for prefix, power in zip(prefixes, powers):
        if abs_value < 10**(power + 3) or prefix == prefixes[-1]:
            return value / 10**power, prefix
        

