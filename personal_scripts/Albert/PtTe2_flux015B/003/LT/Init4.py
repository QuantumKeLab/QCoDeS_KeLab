"""
Init3.py - QCoDeS Experiment Initialization and Analysis Utilities

This module provides a collection of functions and classes to streamline the
initialization, data loading, analysis, and plotting of experiments conducted
using the QCoDeS framework.

Key functionalities include:
- Automatic discovery and initialization of QCoDeS database files.
- SI unit handling and formatting using the `pint` library.
- Data fitting (linear and constant) with SNR calculation.
- Detailed QCoDeS dataset information extraction (metadata, parameters).
- Numerical calculation of differential resistance (dV/dI).
- Generation of 2D heatmaps and other plots using Plotly.
- Utilities for saving plots to various formats.
"""

import os
import sys
import time
import pint
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Retained for colormaps, consider removing if not strictly needed
import plotly.graph_objects as go

# Consider if pyppt is essential or if alternatives like python-pptx are more standard
# import pyppt as ppt

from pprint import pprint
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

# For PDF to image conversion, ensure Poppler is in PATH or specify path
# from pdf2image import convert_from_path
# Example: poppler_path = r"C:\path\to\poppler-xx.yy.zz\bin"

from matplotlib import colormaps # Primarily for 'RdBu' if used directly
from tqdm.notebook import tqdm # For progress bars in Jupyter notebooks
from IPython.display import clear_output # For clearing output in Jupyter notebooks

from scipy import stats, optimize
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths, savgol_filter
# from scipy.optimize import curve_fit # Already imported via optimize
from scipy.interpolate import interp1d

import qcodes as qc
from qcodes.dataset.data_set import DataSet
# from qcodes.dataset.data_export import df_to_xarray_dataset # Removed unused import
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.experiment_container import load_experiment_by_name, new_experiment
from qcodes.dataset.sqlite.database import initialise_or_create_database_at
from qcodes.dataset.plotting import plot_dataset, plot_by_id
from qcodes.parameters import Parameter
# from qcodes.utils.metadata import diff_param_values # Less commonly used directly by users
# from qcodes.instrument.specialized_parameters import ElapsedTimeParameter # If creating measurements programmatically

print(f'Imported all modules, QCoDeS version: {qc.__version__} initialized')

# Global Unit Registry
# Using a global registry is common in scripts, but for larger applications,
# it might be better to pass it around or use a context-specific one.
ureg = pint.UnitRegistry()
ureg.formatter.default_format = '~P'  # Use SI prefix formatting (e.g., "ms", "µV")

def search_and_initialise_db(directory: Optional[str] = None) -> Optional[str]:
    """
    Automatically searches for .db files in the specified or parent directory
    and initializes a selected QCoDeS database.

    Args:
        directory: The directory path to search. If None, uses the parent
                   of the current working directory.

    Returns:
        The path to the initialized database, or None if no DB is found or selected.
    """
    if directory is None:
        directory = os.path.dirname(os.getcwd())
        print(f"No directory provided, searching in parent directory: {directory}")

    db_files: List[str] = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.db'):
                db_files.append(os.path.join(root, file_name))

    if not db_files:
        print(f"No .db files found in {directory} and its subdirectories.")
        return None

    print(f"Found {len(db_files)} .db file(s):")
    for i, db_file in enumerate(db_files):
        print(f"{i + 1}. {db_file}")

    selected_db: Optional[str] = None
    if len(db_files) == 1:
        selected_db = db_files[0]
        print(f"Automatically selected the only .db file: {selected_db}")
    else:
        while True:
            try:
                choice_str = input(f"Enter the number of the .db file to initialize (1-{len(db_files)}): ")
                if not choice_str: # User pressed Enter without input
                    print("No selection made. Aborting database initialization.")
                    return None
                choice = int(choice_str)
                if 1 <= choice <= len(db_files):
                    selected_db = db_files[choice - 1]
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(db_files)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError: # Handle Ctrl+D or similar
                print("Input stream closed. Aborting database initialization.")
                return None


    if selected_db:
        print(f"Initializing database at: {selected_db}")
        initialise_or_create_database_at(selected_db)
        qc.config.core.db_location = selected_db # Ensure QCoDeS config is updated
        return selected_db
    return None

class SI:
    """
    Provides convenient access to SI units via the pint library and
    a formatting utility.
    """
    A = ureg.ampere
    V = ureg.volt
    Ohm = ureg.ohm # Standard spelling
    F = ureg.farad
    H = ureg.henry
    W = ureg.watt
    J = ureg.joule
    s = ureg.second
    m = ureg.meter
    g = ureg.gram # Note: base unit for mass in pint is kg, but g is often used.
    C = ureg.coulomb
    K = ureg.kelvin
    dB = ureg.decibel
    T = ureg.tesla
    Hz = ureg.hertz

    @staticmethod
    def f(value: Union[float, np.ndarray], unit: pint.Unit, precision: int = 2) -> str:
        """
        Formats a numerical value or a NumPy array with its SI unit,
        using compact SI prefix notation.

        Args:
            value: The numerical value or NumPy array to format.
            unit: The pint unit of the value.
            precision: The number of decimal places for formatting.

        Returns:
            A string representation of the value with unit and SI prefix.
        """
        fmt = f".{precision}f~P" # Format string like ".2f~P"

        if isinstance(value, (np.ndarray, list, tuple)): # Added list and tuple
            if isinstance(value, np.ndarray) and value.ndim == 0: # Handle 0-dim arrays
                quantity = float(value) * unit
                return f"{quantity:{fmt}}"
            # For arrays/lists/tuples, format each element
            # This can be verbose for large arrays. Consider alternative representations
            # if this is for display in tables or dense plots.
            return ", ".join([f"{(float(v) * unit):{fmt}}" for v in value])
        else:
            try:
                quantity = float(value) * unit
                return f"{quantity:{fmt}}"
            except TypeError:
                return str(value) # Fallback for non-floatable values

class FitResult:
    """
    Stores the results of a fitting procedure.
    Using dataclasses can make this more concise if Python 3.7+ is standard.
    from dataclasses import dataclass
    @dataclass
    class FitResult:
        fit_name: str
        slope: float
        ...
    """
    def __init__(self, fit_name: str, slope: float, intercept: float,
                 r_value: float, p_value: float, stderr: float,
                 mean_value: float, step_size: Optional[float],
                 noise_std: float, SNR: float):
        self.fit_name = fit_name
        self.slope = slope
        self.intercept = intercept
        self.r_value = r_value
        self.p_value = p_value
        self.stderr = stderr # Standard error of the slope for linregress
        self.mean_value = mean_value
        self.step_size = step_size
        self.noise_std = noise_std
        self.SNR = SNR

@lru_cache(maxsize=32)
def polyfit(fit_name: str, x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Performs a linear regression fit (y = mx + c) on the given data.

    Args:
        fit_name: A descriptive name for the fit.
        x: Independent variable data.
        y: Dependent variable data.

    Returns:
        A FitResult object containing the fit parameters and statistics.
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        warnings.warn(f"Insufficient data for polyfit '{fit_name}'. Returning NaNs.")
        return FitResult(fit_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    step_size = np.mean(np.diff(x)) if len(x) > 1 else None # More robust step_size
    mean_value = np.mean(y)

    # scipy.stats.linregress is good for simple linear fits
    fit = stats.linregress(x, y)
    estimated_signal = fit.slope * x + fit.intercept
    noise = y - estimated_signal
    noise_std = np.std(noise, ddof=2) # ddof=2 because slope and intercept are estimated

    # SNR calculation: Be mindful of its definition.
    # If signal is defined as deviation from mean, or if intercept is meaningful.
    # This definition assumes the linear trend is the "signal".
    signal_power = np.mean(estimated_signal**2)
    # Avoid division by zero or log of zero if noise_std is very small
    SNR = 10 * np.log10(signal_power / (noise_std**2 + 1e-18)) if noise_std > 1e-9 else np.inf

    return FitResult(
        fit_name,
        fit.slope,
        fit.intercept,
        fit.rvalue,
        fit.pvalue,
        fit.stderr, # Standard error of the slope
        mean_value,
        step_size,
        noise_std,
        SNR
    )

@lru_cache(maxsize=32)
def constfit(fit_name: str, x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Fits a constant value (mean) to the y-data.

    Args:
        fit_name: A descriptive name for the fit.
        x: Independent variable data (used for step_size).
        y: Dependent variable data.

    Returns:
        A FitResult object. Slope is 0, intercept is the mean of y.
    """
    if len(x) < 1 or len(y) < 1:
        warnings.warn(f"Insufficient data for constfit '{fit_name}'. Returning NaNs.")
        return FitResult(fit_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    step_size = np.mean(np.diff(x)) if len(x) > 1 else None
    mean_y = np.mean(y)
    intercept = mean_y # For a constant fit, the intercept is the mean
    slope = 0.0

    estimated_signal = np.full_like(y, intercept)
    noise = y - estimated_signal
    noise_std = np.std(noise, ddof=1) # ddof=1 as one parameter (mean) is estimated
    stderr_of_mean = stats.sem(y) if len(y) > 1 else np.nan # Standard error of the mean

    signal_power = np.mean(estimated_signal**2)
    SNR = 10 * np.log10(signal_power / (noise_std**2 + 1e-18)) if noise_std > 1e-9 else np.inf

    return FitResult(
        fit_name,
        slope,
        intercept,
        0.0,  # R-value is not meaningful for a constant fit in this context
        1.0,  # P-value is not meaningful
        stderr_of_mean, # More relevant error for a constant fit
        mean_y,
        step_size,
        noise_std,
        SNR
    )

def print_fit_result(fit_result: FitResult, x_unit: pint.Unit = SI.A, y_unit: pint.Unit = SI.V):
    """
    Prints the formatted results of a fit.

    Args:
        fit_result: The FitResult object to print.
        x_unit: The pint unit of the x-axis data.
        y_unit: The pint unit of the y-axis data.
    """
    print(f"{fit_result.fit_name}:")
    if fit_result.step_size is not None and not np.isnan(fit_result.step_size) :
        print(f"{'Step size:':>20} {SI.f(fit_result.step_size, x_unit)}")
    print(f"{'Mean (y):':>20} {SI.f(fit_result.mean_value, y_unit)}")
    if not np.isnan(fit_result.slope) and fit_result.slope != 0 : # Only print slope if not a constfit
        print(f"{'Slope:':>20} {SI.f(fit_result.slope, y_unit/x_unit)}")
    print(f"{'Intercept:':>20} {SI.f(fit_result.intercept, y_unit)}")
    if not np.isnan(fit_result.r_value) and fit_result.r_value !=0:
        print(f"{'R²:':>20} {fit_result.r_value**2 * 100:.4f}%")
    if not np.isnan(fit_result.p_value) and fit_result.p_value !=1:
        print(f"{'P-value:':>20} {fit_result.p_value:.2e}")
    print(f"{'Std Err (slope/mean):':>20} {SI.f(fit_result.stderr, y_unit/x_unit if fit_result.slope !=0 else y_unit)}")
    print(f"{'Noise SD (residuals):':>20} {SI.f(fit_result.noise_std, y_unit)}")
    print(f"{'SNR:':>20} {fit_result.SNR:.2f} dB\n")


def plot_save(fig: go.Figure, file_id: Union[int, str], save_dir: str = ".",
              save_pdf: bool = False, save_png: bool = True, png_dpi: int = 300) -> None:
    """
    Saves a Plotly figure to specified formats (PDF, PNG).

    Args:
        fig: The Plotly figure object.
        file_id: An identifier for the filename (e.g., run ID).
        save_dir: Directory to save the files. Defaults to current directory.
        save_pdf: Whether to save as PDF.
        save_png: Whether to save as PNG (requires pdf2image and Poppler).
        png_dpi: DPI for the PNG image.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    base_filename = os.path.join(save_dir, str(file_id))

    if save_pdf:
        pdf_filename = f"{base_filename}.pdf"
        try:
            fig.write_image(pdf_filename, scale=2) # Scale can be adjusted
            print(f"Saved PDF: {pdf_filename}")
        except Exception as e:
            print(f"Error saving PDF {pdf_filename}: {e}")
            print("Ensure Orca or Kaleido is installed and configured for Plotly static image export.")

    if save_png:
        # pdf2image is an optional dependency, handle its absence
        try:
            from pdf2image import convert_from_path
            # Ensure poppler_path is set if not in system PATH
            # poppler_path_local = poppler_path # Use global if defined, or set here
            pdf_for_png = f"{base_filename}_temp.pdf"
            png_filename = f"{base_filename}.png"

            try:
                fig.write_image(pdf_for_png, scale=2) # Create a temporary PDF for conversion
                images = convert_from_path(pdf_for_png, dpi=png_dpi) #, poppler_path=poppler_path_local)
                if images:
                    images[0].save(png_filename, "PNG")
                    print(f"Saved PNG: {png_filename}")
                else:
                    print(f"PDF to PNG conversion failed for {pdf_for_png}, no images generated.")
            except Exception as e:
                print(f"Error converting PDF to PNG for {file_id}: {e}")
                print("Ensure pdf2image is installed and Poppler is in your PATH or poppler_path is set.")
            finally:
                if os.path.exists(pdf_for_png):
                    os.remove(pdf_for_png) # Clean up temp PDF
        except ImportError:
            print("pdf2image library not found. Skipping PNG export. Install with 'pip install pdf2image'.")
        except Exception as e: # Catch other potential errors during import or conversion
            print(f"An unexpected error occurred during PNG export setup: {e}")


@lru_cache(maxsize=128) # Increased cache size for datasets
def cached_load_dataset(run_id: int) -> DataSet:
    """
    Loads a QCoDeS dataset by its run_id with caching.

    Args:
        run_id: The ID of the run to load.

    Returns:
        The loaded QCoDeS DataSet.
    """
    print(f"Loading dataset for run_id: {run_id}")
    return qc.load_by_id(run_id)

def get_dataset_info(run_id: int) -> Tuple[str, str, Dict[str, str], datetime, datetime, timedelta, str]:
    """
    Retrieves detailed information about a QCoDeS dataset.

    Args:
        run_id: The ID of the dataset.

    Returns:
        A tuple containing: experiment name, sample name, dictionary of
        parameter names to units, start time, completed time, run duration (timedelta),
        and a human-readable display of the run time.
    """
    dataset = cached_load_dataset(run_id)
    exp_name = dataset.exp_name
    sample_name = dataset.sample_name
    desc = dataset.description

    param_units: Dict[str, str] = {}
    if desc and hasattr(desc, 'interdeps') and hasattr(desc.interdeps, 'parameters'):
        for param_name, param_spec in desc.interdeps.parameters.items():
            param_units[param_name] = param_spec.unit
    else:
        # Fallback for older datasets or different structures if necessary
        # This part might need adjustment based on how metadata was stored if not in interdeps.
        # For example, trying to get units from the parameters themselves if they are in the table
        # but this is less reliable as units might not be consistently stored there.
        print(f"Warning: Could not reliably extract all parameter units for run_id {run_id} from dataset.description.")
        # As a fallback, populate with known parameters from the dataframe if needed, though units might be missing.
        df_param_names = dataset.get_parameters()
        for pspec in df_param_names:
            if pspec.name not in param_units:
                 param_units[pspec.name] = pspec.unit if pspec.unit else "unknown"


    # Time information
    # QCoDeS now stores timestamps as float (Unix epoch time)
    # Convert to datetime objects
    start_timestamp_unix = dataset.run_timestamp_raw
    completed_timestamp_unix = dataset.completed_timestamp_raw

    start_time = datetime.fromtimestamp(start_timestamp_unix) if start_timestamp_unix else datetime.min
    completed_time = datetime.fromtimestamp(completed_timestamp_unix) if completed_timestamp_unix else datetime.min
    
    run_duration = timedelta(seconds=0)
    if start_timestamp_unix and completed_timestamp_unix:
        run_duration = completed_time - start_time
    
    display_run_time = _display_time(run_duration)
    
    # Print basic info
    label_width = 20
    print(f"{'ID':<{label_width}}: {run_id}")
    print(f"{'Type (Experiment)':<{label_width}}: {exp_name}")
    print(f"{'Sample':<{label_width}}: {sample_name}")
    print(f"{'Run time':<{label_width}}: {display_run_time}")
    print(f"{'Parameter units':<{label_width}}:")
    for name, unit in param_units.items():
        print(f"- {name:<{label_width-2}}: {unit}")

    return exp_name, sample_name, param_units, start_time, completed_time, run_duration, display_run_time


def _display_time(duration: timedelta) -> str:
    """
    Formats a timedelta duration into a human-readable string.

    Args:
        duration: The timedelta object representing the duration.

    Returns:
        A string like "X hr Y min Z sec".
    """
    total_seconds = int(duration.total_seconds())

    if total_seconds < 0: # Should not happen for run duration
        return "N/A (invalid duration)"
    if total_seconds < 60:
        return f"{total_seconds} sec"
    
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} sec" if seconds else f"{minutes} min"
    
    hours, minutes = divmod(minutes, 60)
    return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"


def info_df(run_id: int) -> Tuple[List[str], str, Dict[str, Any], pd.DataFrame]:
    """
    Loads a dataset, extracts parameter information, calculates dV/dI if applicable,
    and returns key information along with the data as a Pandas DataFrame.

    Args:
        run_id: The ID of the QCoDeS dataset.

    Returns:
        A tuple containing:
        - List of setpoint parameter names.
        - Name of the dependent parameter.
        - Dictionary with info about each parameter (points, type, range, step).
        - Pandas DataFrame of the dataset, potentially with an added 'dV_dI' column.
    """
    dataset = cached_load_dataset(run_id)
    param_data = dataset.get_parameter_data() # This is a dictionary of dictionaries of numpy arrays

    # The structure of param_data is {dependent_param_name: {param_name: data_array}}
    # The first key is typically the "primary" dependent parameter if multiple exist.
    # We assume one primary dependent parameter for simplicity here.
    if not param_data:
        raise ValueError(f"Dataset {run_id} appears to be empty or has no loadable parameters.")

    dependent_param_name = next(iter(param_data))
    # All parameters measured alongside this dependent one
    all_param_names_for_dep = list(param_data[dependent_param_name].keys())

    setpoint_param_names = [p_name for p_name in all_param_names_for_dep if p_name != dependent_param_name]

    param_info: Dict[str, Dict[str, Any]] = {}

    print(f"\nNumber of points for each parameter in dataset {run_id}:")
    dep_data_array = param_data[dependent_param_name][dependent_param_name]
    print(f"- {dependent_param_name} (dependent): {len(dep_data_array)} points")
    param_info[dependent_param_name] = {
        'points': len(dep_data_array),
        'type': 'dependent'
    }

    for sp_name in setpoint_param_names:
        sp_data_array = param_data[dependent_param_name][sp_name]
        unique_values = np.unique(sp_data_array)
        num_unique = len(unique_values)
        start_val, end_val = (unique_values[0], unique_values[-1]) if num_unique > 0 else (np.nan, np.nan)
        step_size = np.mean(np.diff(unique_values)) if num_unique > 1 else None

        param_info[sp_name] = {
            'unique_points': num_unique,
            'start': start_val,
            'end': end_val,
            'step_size': step_size,
            'type': 'setpoint'
        }
        step_str = f"{step_size:.2e}" if step_size is not None else "N/A"
        print(f"- {sp_name:<16} (setpoint): {num_unique} unique points, from {start_val:.2e} to {end_val:.2e}, step size: {step_str}")

    # Convert to Pandas DataFrame
    # QCoDeS' to_pandas_dataframe() might create a MultiIndex if there are multiple setpoints.
    # We'll use the xarray intermediate representation for more control if needed,
    # or handle MultiIndex directly.
    try:
        # This method is generally robust for converting to a flat DataFrame
        xds = dataset.to_xarray_dataset()
        df = xds.to_dataframe().reset_index()
    except Exception as e:
        print(f"Failed to convert dataset {run_id} to DataFrame via xarray: {e}. Trying legacy method.")
        # Fallback to older method, which might handle some legacy datasets differently
        df = dataset.to_pandas_dataframe().reset_index()


    # Attempt to find voltage and current parameters for dV/dI
    # This relies on naming conventions. More robust would be to store metadata about parameter types.
    voltage_param_name = dependent_param_name # Assuming the primary dependent is voltage
    
    # Try to find a current parameter among setpoints
    current_param_name: Optional[str] = None
    possible_current_names = ['curr', 'current', 'I', 'bias_current'] # Common names for current
    for sp_name in setpoint_param_names:
        if any(cn.lower() in sp_name.lower() for cn in possible_current_names):
            current_param_name = sp_name
            break
    
    if current_param_name and voltage_param_name in df.columns:
        print(f"\nCalculating dV/dI using voltage: {voltage_param_name} and current: {current_param_name}")

        # Identify other setpoint parameters to group by
        other_setpoint_names = [p_name for p_name in setpoint_param_names if p_name != current_param_name]

        if other_setpoint_names:
            df['dV_dI'] = np.nan # Initialize column
            grouped = df.groupby(other_setpoint_names)
            
            # tqdm for progress bar if many groups
            for _, group_df_indices in tqdm(grouped.groups.items(), desc="Calculating dV/dI per group"):
                group_df = df.loc[group_df_indices].sort_values(by=current_param_name)
                if len(group_df[current_param_name]) > 1: # Need at least 2 points for gradient
                    # Ensure current values are monotonically increasing for gradient
                    # and handle potential duplicate current values if any (though unlikely for a sweep)
                    unique_currents, unique_indices = np.unique(group_df[current_param_name], return_index=True)
                    if len(unique_currents) > 1:
                        voltage_at_unique_currents = group_df[voltage_param_name].iloc[unique_indices].values
                        # np.gradient is sensitive to noise. Consider Savitzky-Golay filter first if data is noisy.
                        # Example: voltage_filtered = savgol_filter(voltage_at_unique_currents, window_length=5, polyorder=2)
                        # dv_di = np.gradient(voltage_filtered, unique_currents)
                        dv_di = np.gradient(voltage_at_unique_currents, unique_currents)
                        
                        # Map back results carefully if there were duplicates or unsorted data
                        # This simple assignment works if group_df was already sorted and current_param had unique values within group
                        df.loc[group_df.index, 'dV_dI'] = np.interp(group_df[current_param_name], unique_currents, dv_di)
                    else:
                        df.loc[group_df.index, 'dV_dI'] = np.nan # Not enough unique points
                else:
                    df.loc[group_df.index, 'dV_dI'] = np.nan # Not enough points
            print("Added 'dV_dI' column to DataFrame, calculated per group.")
        elif len(df[current_param_name]) > 1 : # No other setpoints, calculate dV/dI directly
            df = df.sort_values(by=current_param_name) # Ensure sorted by current
            unique_currents, unique_indices = np.unique(df[current_param_name], return_index=True)
            if len(unique_currents) > 1:
                voltage_at_unique_currents = df[voltage_param_name].iloc[unique_indices].values
                dv_di = np.gradient(voltage_at_unique_currents, unique_currents)
                df['dV_dI'] = np.interp(df[current_param_name], unique_currents, dv_di)
                print("Added 'dV_dI' column to DataFrame.")
            else:
                 df['dV_dI'] = np.nan
                 print("Not enough unique current points to calculate dV/dI.")
        else:
            df['dV_dI'] = np.nan
            print("Not enough data points to calculate dV/dI.")
    else:
        print("\nCould not identify distinct voltage and current parameters for dV/dI calculation.")

    print("\nDataFrame Info:")
    df.info()
    print("\nDataFrame Description:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())

    return setpoint_param_names, dependent_param_name, param_info, df


def plot_heatmaps(run_id: int) -> Tuple[Optional[go.Figure], Optional[go.Figure], Optional[go.Figure], Optional[pd.DataFrame]]:
    """
    Generates and displays heatmaps for voltage and differential resistance (dV/dI)
    from a QCoDeS dataset. Also attempts to plot Ic vs the first setpoint parameter.

    Args:
        run_id: The ID of the QCoDeS dataset.

    Returns:
        A tuple containing:
        - fig1 (Plotly Figure or None): Heatmap of the primary dependent parameter.
        - fig2 (Plotly Figure or None): Heatmap of dV/dI, if calculated.
        - fig3 (Plotly Figure or None): Plot of Ic vs. the first setpoint, if applicable.
        - Ic_df (DataFrame or None): DataFrame containing critical current data, if calculated.
    """
    try:
        exp_name, sample_name, param_units, start_time, _, _, display_run_time = get_dataset_info(run_id)
        setpoint_params, dependent_param, param_info, df = info_df(run_id)
    except Exception as e:
        print(f"Error getting dataset info or DataFrame for run_id {run_id}: {e}")
        return None, None, None, None

    if len(setpoint_params) < 2:
        print(f"Dataset {run_id} does not have at least two setpoint parameters for a 2D heatmap. Skipping heatmaps.")
        # Try to make a 1D plot if possible
        if len(setpoint_params) == 1 and dependent_param in df.columns and setpoint_params[0] in df.columns:
            fig1 = go.Figure(data=go.Scatter(x=df[setpoint_params[0]], y=df[dependent_param], mode='lines+markers'))
            fig1.update_layout(
                title=f"<b>#{run_id} | {exp_name} |</b> {sample_name}<br><sup>Run Time: {display_run_time}</sup>",
                xaxis_title=f"{setpoint_params[0]} ({param_units.get(setpoint_params[0], '')})",
                yaxis_title=f"{dependent_param} ({param_units.get(dependent_param, '')})"
            )
            return fig1, None, None, None
        return None, None, None, None

    # For heatmaps, we need at least two setpoint parameters.
    # Let's assume the first two setpoint_params are for x and y axes of the heatmap.
    # QCoDeS typically orders them from "fastest" to "slowest" swept.
    # For heatmaps, often x is the inner loop (fastest) and y is the outer loop (slower).
    # The `to_xarray_dataset` and then `to_dataframe` usually flattens this,
    # so we might need to pivot if the data isn't already gridded.
    # However, plotly's heatmap can sometimes infer the grid from x, y, z columns.

    x_ax_param = setpoint_params[0] # e.g., 'y_field' in the notebook
    y_ax_param = setpoint_params[1] # e.g., 'appl_current' in the notebook
    
    # Ensure data is sorted for consistent plotting, especially if pivoting is needed
    # or if Plotly relies on implicit ordering for heatmaps from 1D arrays.
    df_sorted = df.sort_values(by=[y_ax_param, x_ax_param])


    # Annotations for plot metadata
    # Ensure start_time is available from get_dataset_info
    plot_annotations = []
    if start_time != datetime.min: # Check if start_time is valid
         plot_annotations.append(
            go.layout.Annotation(
                text=f"{start_time.strftime('%Y-%m-%d %H:%M')} | Duration: {display_run_time}",
                align='left', xref='paper', yref='paper', x=0, y=1.02, showarrow=False,
                font=dict(size=10)
            )
        )

    if x_ax_param in param_info and param_info[x_ax_param]['type'] == 'setpoint':
        points = param_info[x_ax_param]['unique_points']
        step = param_info[x_ax_param]['step_size']
        step_str = f"{step:.2e}" if step is not None and not np.isnan(step) else "N/A"
        plot_annotations.append(go.layout.Annotation(
            text=f"{points} pts <br>step={step_str}",
            align='right', xref='paper', yref='paper', x=1, y=-0.06, showarrow=False, font=dict(size=10)
        ))
    if y_ax_param in param_info and param_info[y_ax_param]['type'] == 'setpoint':
        points = param_info[y_ax_param]['unique_points']
        step = param_info[y_ax_param]['step_size']
        step_str = f"{step:.2e}" if step is not None and not np.isnan(step) else "N/A"
        plot_annotations.append(go.layout.Annotation(
            text=f"{points} pts <br>step={step_str}",
            align='right', xref='paper', yref='paper', x=-0.08, y=1, showarrow=False, textangle=-90, font=dict(size=10)
        ))
        
    # --- Voltage Heatmap (fig1) ---
    try:
        # Check if data is already gridded (all x for a given y, then next y, etc.)
        # Plotly heatmap can often handle ungridded x, y, z if they correspond.
        fig1 = go.Figure(data=go.Heatmap(
            x=df_sorted[x_ax_param],
            y=df_sorted[y_ax_param],
            z=df_sorted[dependent_param],
            colorscale='RdBu_r', # Reversed RdBu is often preferred for V or dV/dI
            colorbar=dict(title=dict(text=f"{dependent_param} ({param_units.get(dependent_param, '')})", side='right'))
        ))
        fig1.update_layout(
            title=f"<b>#{run_id} | {exp_name} |</b> {sample_name}",
            xaxis_title=f"{x_ax_param} ({param_units.get(x_ax_param, '')})",
            yaxis_title=f"{y_ax_param} ({param_units.get(y_ax_param, '')})",
            annotations=plot_annotations,
            width=800, height=700 # Adjust as needed
        )
    except Exception as e:
        print(f"Error creating voltage heatmap (fig1) for run_id {run_id}: {e}")
        fig1 = None

    # --- dV/dI Heatmap (fig2) ---
    fig2 = None
    dvdi_col_name = 'dV_dI' # Standardized name from info_df
    if dvdi_col_name in df_sorted.columns and not df_sorted[dvdi_col_name].isnull().all(): # Check if column exists and is not all NaN
        try:
            # Determine robust z-limits for dV/dI to handle outliers/infinities
            dvdi_data_finite = df_sorted[dvdi_col_name][np.isfinite(df_sorted[dvdi_col_name])]
            if not dvdi_data_finite.empty:
                zmin_dvdi = dvdi_data_finite.quantile(0.01) # Percentile clipping
                zmax_dvdi = dvdi_data_finite.quantile(0.99)
                # If min and max are too close, expand the range slightly
                if np.isclose(zmin_dvdi, zmax_dvdi):
                    z_center = zmin_dvdi # or zmax_dvdi
                    z_range_half = 0.1 * abs(z_center) if not np.isclose(z_center, 0) else 0.1
                    zmin_dvdi = z_center - z_range_half
                    zmax_dvdi = z_center + z_range_half
                    if np.isclose(zmin_dvdi, zmax_dvdi): # Still close (e.g. all zeros)
                         zmin_dvdi = -1.0
                         zmax_dvdi = 1.0
            else: # All values are non-finite or column was all NaN initially
                zmin_dvdi, zmax_dvdi = None, None


            fig2 = go.Figure(data=go.Heatmap(
                x=df_sorted[x_ax_param],
                y=df_sorted[y_ax_param],
                z=df_sorted[dvdi_col_name],
                colorscale='RdBu_r',
                zmin=zmin_dvdi,
                zmax=zmax_dvdi,
                colorbar=dict(title=dict(text=f"{dvdi_col_name} ({param_units.get(dvdi_col_name, SI.Ohm)})", side='right')) # Assume Ohm if unit not found
            ))
            fig2.update_layout(
                title=f"<b>#{run_id} | {exp_name} |</b> {sample_name} - dV/dI",
                xaxis_title=f"{x_ax_param} ({param_units.get(x_ax_param, '')})",
                yaxis_title=f"{y_ax_param} ({param_units.get(y_ax_param, '')})",
                annotations=plot_annotations,
                width=800, height=700
            )
        except Exception as e:
            print(f"Error creating dV/dI heatmap (fig2) for run_id {run_id}: {e}")
            fig2 = None
    else:
        print(f"'{dvdi_col_name}' column not found or is all NaN in DataFrame for run_id {run_id}. Skipping dV/dI heatmap.")

    # --- Ic vs. first setpoint parameter (fig3) ---
    fig3 = None
    Ic_df = None
    # Attempt to find current parameter name again, as it's not passed directly to plot_heatmaps
    current_param_name_local: Optional[str] = None
    possible_current_names = ['curr', 'current', 'I', 'bias_current']
    # y_ax_param is often current in these types of 2D scans (e.g. field vs current)
    if any(cn.lower() in y_ax_param.lower() for cn in possible_current_names):
        current_param_name_local = y_ax_param
    else: # Fallback: check other setpoints if y_ax_param wasn't current
        for sp_name in setpoint_params:
            if any(cn.lower() in sp_name.lower() for cn in possible_current_names):
                current_param_name_local = sp_name
                break

    if dvdi_col_name in df_sorted.columns and not df_sorted[dvdi_col_name].isnull().all() and current_param_name_local:
        print(f"\nAttempting to extract and plot Ic based on '{dvdi_col_name}' peaks vs '{x_ax_param}' using '{current_param_name_local}' as current...")
        Ic_values = []
        setpoint_values_for_Ic = []

        try:
            # Ensure current_param_name_local is the one swept against x_ax_param
            # The grouping should be by the parameter that is NOT the current sweep axis
            # If x_ax_param is field and y_ax_param is current, group by x_ax_param
            grouping_param_for_Ic = x_ax_param
            current_sweep_param_for_Ic = y_ax_param # This should be the current_param_name_local

            if current_param_name_local != y_ax_param:
                # This case is more complex: if the identified current_param_name_local
                # is not one of the primary heatmap axes, the logic needs adjustment.
                # For now, assume y_ax_param is the current sweep for Ic extraction.
                # If not, this Ic extraction might be incorrect.
                print(f"Warning: Identified current parameter '{current_param_name_local}' is not the y-axis of heatmap ('{y_ax_param}'). Ic extraction might be inaccurate.")


            for x_val, group in tqdm(df_sorted.groupby(grouping_param_for_Ic), desc=f"Extracting Ic vs {grouping_param_for_Ic}"):
                group_sorted_by_current = group.sort_values(by=current_sweep_param_for_Ic)
                
                positive_current_group = group_sorted_by_current[group_sorted_by_current[current_sweep_param_for_Ic] > 0]
                if not positive_current_group.empty and not positive_current_group[dvdi_col_name].isnull().all():
                    # Use a more robust peak height, e.g., mean + std or a fraction of max
                    peak_height_threshold = positive_current_group[dvdi_col_name].mean() + 0.5 * positive_current_group[dvdi_col_name].std()
                    peaks, properties = find_peaks(positive_current_group[dvdi_col_name].values, height=peak_height_threshold)
                    
                    if len(peaks) > 0:
                        # Could add logic to pick the most prominent peak if multiple exist
                        ic_val_pos = positive_current_group[current_sweep_param_for_Ic].iloc[peaks[0]]
                        Ic_values.append(ic_val_pos)
                        setpoint_values_for_Ic.append(x_val)
            
            if Ic_values:
                Ic_df = pd.DataFrame({
                    grouping_param_for_Ic: setpoint_values_for_Ic,
                    'Ic': Ic_values
                })
                fig3 = go.Figure(data=go.Scatter(
                    x=Ic_df[grouping_param_for_Ic],
                    y=Ic_df['Ic'],
                    mode='lines+markers'
                ))
                # Use only the first annotation (timestamp) for Ic plot if available
                ic_plot_annotations = plot_annotations[:1] if plot_annotations else []
                fig3.update_layout(
                    title=f"<b>#{run_id} | {exp_name} |</b> {sample_name} - Estimated Ic",
                    xaxis_title=f"{grouping_param_for_Ic} ({param_units.get(grouping_param_for_Ic, '')})",
                    yaxis_title=f"Critical Current Ic ({param_units.get(current_sweep_param_for_Ic, '')})",
                    annotations=ic_plot_annotations,
                    width=800, height=600
                )
                print("Generated Ic plot (fig3).")
            else:
                print("No significant Ic values extracted to plot.")
        except Exception as e:
            print(f"Error during Ic extraction/plotting for run_id {run_id}: {e}")
            fig3 = None
            Ic_df = None
    else:
        if not (dvdi_col_name in df_sorted.columns and not df_sorted[dvdi_col_name].isnull().all()):
            print(f"Skipping Ic plot: '{dvdi_col_name}' column not found or all NaN.")
        if not current_param_name_local:
             print(f"Skipping Ic plot: Could not identify current parameter for Ic extraction.")
            
    return fig1, fig2, fig3, Ic_df


# Functions related to `generate_points_for_parameter`, `align_Ic_scans_and_find_max`,
# `plot_Ic_vs_B`, `plot_calibrated_Ic_vs_B`, `plot_Ic_vs_angle`,
# and the `main` function with its `CONFIG` would typically be part of a specific
# analysis script (like your notebook) rather than a general Init.py,
# unless they are very generic helper functions.
# If they are generic, they should be documented and typed similarly.

# Example of how the notebook functions might be structured if kept in Init3.py
# (This is a placeholder, as their full context is in the notebook)

DEFAULT_CONFIG = {
    "crit_dVdI": 1e-1,  # V/A or Ohm, example value
    "crit_V": 1e-6,     # Volts, example value
    "align_window_factor": 0.2,
    "poly_order_sg_filter": 2,
    "window_length_sg_filter": 11, # Must be odd
    "peak_prominence_factor": 0.1,
    "x_axis_label": "Magnetic Field (T)", # Example
    "offset_factor": 1.0 # Example
}

def generate_points_for_parameter(
    MagCenter: float,
    RangeOuter: float,
    RangeInner: float,
    RangeCenter: float,
    step_outer: float,
    step_inner: float,
    step_center: float
) -> np.ndarray:
    """
    Generates a non-uniformly spaced array of points for a parameter sweep,
    typically for magnetic field, with denser sampling around a central point.

    Args:
        MagCenter: The center of the sweep.
        RangeOuter: The total range of the sweep on one side of MagCenter.
        RangeInner: The range for denser sampling on one side of MagCenter,
                    inside RangeOuter.
        RangeCenter: The range for the densest sampling around MagCenter,
                     inside RangeInner.
        step_outer: Step size in the outermost regions.
        step_inner: Step size in the intermediate regions.
        step_center: Step size in the central, densest region.

    Returns:
        A NumPy array of points for the parameter sweep.
    """
    # Define the relative boundaries of the 5 scan segments from MagCenter
    # Ensure ranges are positive and correctly ordered
    RangeOuter = abs(RangeOuter)
    RangeInner = abs(RangeInner)
    RangeCenter = abs(RangeCenter)

    if not (RangeOuter >= RangeInner >= RangeCenter >= 0):
        raise ValueError("Ranges must be positive and ordered: Outer >= Inner >= Center >= 0")
    if not (step_outer > 0 and step_inner > 0 and step_center > 0):
        raise ValueError("Step sizes must be positive.")

    edges = np.array([
        -RangeOuter, -RangeInner, -RangeCenter, RangeCenter, RangeInner, RangeOuter
    ])
    
    step_sizes = np.array([
        step_outer, step_inner, step_center, step_inner, step_outer
    ])

    absolute_edges = MagCenter + edges
    
    all_segment_points = []
    for i in range(len(step_sizes)):
        segment_start, segment_end = absolute_edges[i], absolute_edges[i+1]
        
        # Ensure start is less than or equal to end for np.linspace
        # This also handles zero-length segments correctly if start == end
        current_step = step_sizes[i]
        if segment_start > segment_end: # Should ideally not happen if ranges are defined correctly
            segment_start, segment_end = segment_end, segment_start # Swap
        
        if np.isclose(segment_start, segment_end):
            segment_points = np.array([segment_start])
        else:
            # Calculate number of points for linspace: (length / step) + 1
            # Ensure at least 2 points if start != end, even if step is large
            n_pts_in_segment = max(2, int(np.round(abs(segment_end - segment_start) / current_step)) + 1)
            segment_points = np.linspace(segment_start, segment_end, n_pts_in_segment)
        
        if i == 0:
            all_segment_points.append(segment_points)
        else:
            # Exclude the first point of subsequent segments if it's same as last of previous
            if not np.isclose(segment_points[0], all_segment_points[-1][-1]):
                 all_segment_points.append(segment_points)
            else:
                 all_segment_points.append(segment_points[1:])


    # Concatenate all points from all segments
    if not all_segment_points: # Should not happen with valid inputs
        return np.array([MagCenter])

    scan_points = np.concatenate(all_segment_points)
    
    # Round to a reasonable precision to handle floating point issues before unique
    # The precision (e.g., 9) should be chosen based on instrument capabilities/desired resolution
    return np.unique(np.round(scan_points, 9))


