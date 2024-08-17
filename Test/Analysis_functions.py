# %%
import os
import gc
import cairosvg
import numpy as np
import qcodes as qc
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from tqdm import tqdm
from Analysis_functions import *
from scipy.signal import find_peaks
from qcodes.parameters import Parameter
from IPython.display import clear_output
from qcodes.dataset import (
    Measurement, initialise_or_create_database_at,
    load_or_create_experiment, plot_dataset)
def extract_sample_and_number(sample_name):
    # Define the pattern for extracting the sample and number
    pattern = r'(?P<sample>.*?)(?P<number>\d+)$'

    # Use regular expression to extract the sample and number
    match = re.match(pattern, sample_name)

    if match:
        sample = match.group('sample')
        number = int(match.group('number'))
        return sample, number
    else:
        return None, None


def extract_sample_junction(sample_name):
    """Extracts the complete sample junction from the provided sample name.

    Args:
      sample_name: The name of the sample containing the junction information.

    Returns:
      The extracted sample junction, or None if not found.
    """

    delimiter = "_J"
    if delimiter in sample_name:
        # +3 to include "_J"
        return sample_name[:sample_name.find(delimiter) + 3]
    else:
        return None


def get_detaset_info(dataset):
    run_id = dataset.captured_run_id
    exp_name = dataset.exp_name
    sample_name = dataset.sample_name
    name = dataset.name
    description = dataset.description
    sample_junction = extract_sample_junction(sample_name)
    return run_id, exp_name, sample_name, sample_junction, name, description


def IV_Mag_plot(run_id, save=False, log_lower_limit=0):
    # Load the dataset
    dataset = qc.load_by_id(run_id)
    # Get the dataset info
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df = dataset.to_pandas_dataframe().reset_index()
    # Plot the dataset
    plot_dataset(dataset)
    para_list = dataset.parameters.split(",")

    fig1 = go.Figure(data=go.Heatmap(
        x=df[para_list[0]]*1e-9,
        y=df[para_list[1]]*1e6,
        z=df[para_list[2]]*1e3,
        colorscale='RdBu',
        colorbar=dict(
            title="Voltage(mV)", titleside='top')),
        layout=dict(
            title=f"#{run_id} JJ2 IV-F RF Power@20dBm, B_y@0.1mT",
            xaxis_title="RF Frequency (GHz)",
            yaxis_title="Current (μA)",
            height=720, width=2560,
            margin=dict(l=10, r=10, t=50, b=10)

    ))
    # fig1.show()

    df['differential_voltage'] = (
        df[para_list[2]].shift(-1) -
        df[para_list[2]].shift(1)) / (2 * (df[para_list[1]].shift(-1) - df[para_list[1]]))
    fig2 = go.Figure(data=go.Heatmap(
        x=df[para_list[0]],
        y=df[para_list[1]],
        z=df['differential_voltage'],
        colorscale='RdBu',
        colorbar=dict(
            title='dV/dI', titleside='top')),
        layout=dict(
            title=f"#{run_id} dV/dI {sample_name}",
            xaxis_title=para_list[0],
            yaxis_title=para_list[1],
            height=800, width=800
    ))
    # fig2.show()
    fig3 = go.Figure(data=go.Heatmap(
        x=df[para_list[0]],
        y=df[para_list[1]],
        z=np.log10(df['differential_voltage'].replace(
            0, 1e-20).clip(lower=log_lower_limit)),
        colorscale='RdBu',
        colorbar=dict(
            title='dV/dI(Log)', titleside='top')),
        layout=dict(
            title=f"#{run_id} dV/dI {sample_name}",
            xaxis_title=para_list[0],
            yaxis_title=para_list[1],
            height=720, width=2560,
            margin=dict(l=10, r=10, t=50, b=10)
    ))
    # fig3.show()
    
    fig4 = go.Figure(data=go.Heatmap(
        x=df[para_list[0]],
        y=df[para_list[2]],
        z=np.log10(df['differential_voltage'].replace(
            0, 1e-20).clip(lower=log_lower_limit)),
        colorscale='RdBu',
        colorbar=dict(
            title='dV/dI(Log)', titleside='top')),
        layout=dict(
            title=f"#{run_id} dV/dI {sample_name}",
            xaxis_title=para_list[0],
            yaxis_title=para_list[2],
            height=720, width=720
    ))
    import plotly.subplots as subplots

    # ...

    # Create figure layouts
    fig1_layout = fig1.layout
    fig2_layout = fig2.layout
    fig3_layout = fig3.layout

    # Create a grid of subplots
    fig = subplots.make_subplots(rows=1, cols=3, subplot_titles=(
        'IV Magnitude', 'dV/dI', 'dV/dI(Log)'))

    # Add traces to the subplots
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    fig.add_trace(fig3.data[0], row=1, col=3)

    # Update the layout with titles and dimensions
    fig.layout.update(title_text=f"#{run_id} {exp_name} {sample_name}", height=800, width=2400)

    # Update xaxis and yaxis properties individually
    fig.update_xaxes(fig1_layout.xaxis, row=1, col=1)
    fig.update_yaxes(fig1_layout.yaxis, row=1, col=1)
    fig.update_xaxes(fig2_layout.xaxis, row=1, col=2)
    fig.update_yaxes(fig2_layout.yaxis, row=1, col=2)
    fig.update_xaxes(fig3_layout.xaxis, row=1, col=3)
    fig.update_yaxes(fig3_layout.yaxis, row=1, col=3)
    fig.data[0].colorbar.x = 0.288  # Adjust x position of the first colorbar
    fig.data[1].colorbar.x = 0.642
    fig.data[2].colorbar.x = 1

    # Show the combined figure
    # fig.show()
    fig1.show()
    # fig2.show()
    fig3.show()
    # fig4.show()

    if save == True:
        pio.write_image(fig1, rf"D:\data\Albert\103\Plot\103_{run_id}_IV_Mag.png")
        pio.write_image(fig2, rf"D:\data\Albert\103\Plot\103_{run_id}_dVdI.png")
        pio.write_image(fig3, rf"D:\data\Albert\103\Plot\103_{run_id}_dVdI(Log).png")


def IV_RF_f_linecut(run_id, target_frequencies=[5.0, 6.0]):
    dataset = qc.load_by_id(run_id)
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df = dataset.to_pandas_dataframe().reset_index()
    para_list = dataset.parameters.split(",")

    fig = go.Figure()

    for target_frequency in target_frequencies:
        # Find the closest value to the target frequency
        closest_frequency = df[para_list[0]].iloc[(
            df[para_list[0]]-target_frequency*1e9).abs().argsort()[:1]].values[0]

        # Filter the data for the closest frequency
        SGS_cut = df[df[para_list[0]] == closest_frequency]

        # Linecut plot
        fig.add_trace(go.Scatter(x=SGS_cut[para_list[1]], y=SGS_cut[para_list[2]], mode='lines', name=f'{closest_frequency*1e-9:.4f} GHz'))

    fig.update_layout(
        title=f"#{run_id} {exp_name} {sample_name} <br> Linecut",
        xaxis_title=para_list[1],
        yaxis_title=para_list[2],
        height=800, width=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5
        ))

    fig.show()


def IV_RF_PD_linecut(run_id, target_powers=[-10.0, 0.0]):
    dataset = qc.load_by_id(run_id)
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df = dataset.to_pandas_dataframe().reset_index()
    para_list = dataset.parameters.split(",")

    fig = go.Figure()

    for target_power in target_powers:
        # Find the closest value to the target power
        closest_power = df[para_list[0]].iloc[(
            df[para_list[0]]-target_power).abs().argsort()[:1]].values[0]

        # Filter the data for the closest frequency
        SGS_cut = df[df[para_list[0]] == closest_power]

        # Linecut plot
        fig.add_trace(go.Scatter(
            x=SGS_cut[para_list[1]], y=SGS_cut[para_list[2]], mode='lines', name=f'{closest_power:.1f} dBm'))

    fig.update_layout(
        title=f"#{run_id} {exp_name} {sample_name} <br> Linecut",
        xaxis_title=para_list[1],
        yaxis_title=para_list[2],
        height=800, width=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5
        ))

    fig.show()

def svg_to_png(svg_path, png_path, dpi=3000):
    cairosvg.svg2png(url=svg_path, output_width=dpi, write_to=png_path)

def IV(run_id=1):
    dataset = qc.load_by_id(run_id)
    R = get_R_from_IV(run_id)
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df = dataset.to_pandas_dataframe().reset_index()
    para_list = dataset.parameters.split(",")

    # Create a Plotly trace
    trace = go.Scatter(x=df[para_list[0]], y=df[para_list[1]], mode='lines', name='Voltage vs. Current')

    # Create layout with micro-scale y-axis
    layout = go.Layout(title=f'<b>#{run_id} {exp_name}</b> {sample_name}, R_fit={R:.4f} Ω', title_font=dict(size=28),
                       xaxis=dict(title=para_list[0], tickfont=dict(size=16),title_font=dict(size=20)),
                       yaxis=dict(title=f"{para_list[1]}", tickfont=dict(size=16),title_font=dict(size=20)), 
                       width=1080, height=960,
                       margin=dict(l=10, r=10, t=50, b=10))
    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    pio.write_image(fig, r"Test/104_IV_{}.svg".format(run_id))
    # svg_to_png(r"Test/104_IV_{}.svg".format(run_id), r"Test/104_IV_{}.jpg".format(run_id), 2160)
    # Plot the figure
    fig.show()

def IV_diff(run_id=1):
    dataset = qc.load_by_id(run_id)
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df = dataset.to_pandas_dataframe().reset_index()
    para_list = dataset.parameters.split(",")

    current = df[para_list[0]].tolist()
    current_array = np.array(current)
    voltage = df[para_list[1]].tolist()

    differential_voltage = np.diff(voltage) / np.diff(current)
    current = current[:-1]

    peaks, _ = find_peaks(differential_voltage)

    # Extract peak current and peak differential voltage values
    peak_current = [current[i] for i in peaks]
    peak_voltage_diff = [differential_voltage[i] for i in peaks]

    # Sort the peaks based on their values
    sorted_peak_indices = np.argsort(peak_voltage_diff)[::-1]  # Sort in descending order
    top_two_peak_indices = sorted_peak_indices[:2]  # Select the top two indices

    # Extract the top two peaks
    top_two_peak_current = [peak_current[i] for i in top_two_peak_indices]
    top_two_peak_voltage_diff = [peak_voltage_diff[i] for i in top_two_peak_indices]

    # Calculate the center points of the top two peaks
    center_points = [(top_two_peak_current[i] + top_two_peak_current[i+1]) / 2 for i in range(len(top_two_peak_current) - 1)]

    # Calculate the difference between the top two biggest peaks
    peak_difference = abs(top_two_peak_current[0] - top_two_peak_current[1])
    current_array = np.array(current)
    indices = [np.argmin(np.abs(current_array - cp)) for cp in center_points]
    # indices = [np.argmin(np.abs(current  - cp)) for cp in center_points]


    # Determine the appropriate unit for current
    if peak_difference >= 1e-3:
        unit = "mA"
        conversion_factor = 1e3
    elif peak_difference >= 1e-6:
        unit = "μA"
        conversion_factor = 1e6
    else:
        unit = "nA"
        conversion_factor = 1e9

    # Determine the appropriate unit for current
    if peak_difference >= 1e-3:
        unit = "mA"
    elif peak_difference >= 1e-6:
        unit = "μA"
    elif peak_difference >= 1e-9:
        unit = "nA"
    else:
        unit = "pA"

    # Create a Plotly trace for the differential
    trace_diff = go.Scatter(x=[c * conversion_factor for c in current], y=differential_voltage, mode='lines', name='dV/dI')
    # Add markers for the top two peaks
    trace_top_two_peaks = go.Scatter(x=[p * conversion_factor for p in top_two_peak_current], y=top_two_peak_voltage_diff, mode='markers', marker=dict(color='red'), name='Top Two Peaks')
    # Add markers for the center points
    trace_center_points = go.Scatter(x=[cp * conversion_factor for cp in center_points], y=[0]*len(center_points), mode='markers', marker=dict(color='blue', symbol='cross'), name='Center Points')

    # Create layout
    layout = go.Layout(title=f"<b>#{run_id} I-dV/dI</b> {sample_name}",title_font=dict(size=28),
                    xaxis=dict(title=f'Current ({unit})',tickfont=dict(size=16),title_font=dict(size=20)),
                    yaxis=dict(title='dV/dI',tickfont=dict(size=16),title_font=dict(size=20)),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.07, xanchor="right", x=1),
                    width=1080, height=960,
                    margin=dict(l=10, r=10, t=50, b=10))

    # Create the figure
    fig = go.Figure(data=[trace_diff, trace_top_two_peaks, trace_center_points], layout=layout)

    # Add annotation for the current value of center points
    for i, center_point in enumerate(center_points):
        fig.add_annotation(
            x=center_point * conversion_factor,
            y=1.5 * min(differential_voltage)+20,
            text=f'Center Point: {center_point * conversion_factor:.1f}{unit} <br> Peak difference: {peak_difference*conversion_factor:.1f} {unit}',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            ax=-100,
            ay=-200 - 30 * i
        )

    # Plot the figure
    fig.show()
    pio.write_image(fig, r"Test/104_IV_diff_{}.svg".format(run_id))


def get_R_from_IV(dataid):
    dataset = qc.load_by_id(dataid)

    voltage = dataset.get_parameter_data(
    )['meas_voltage_K2']['meas_voltage_K2']
    current = dataset.get_parameter_data()['meas_voltage_K2']['appl_current']
    fit_values = np.polyfit(current, voltage, 1)

    return fit_values[0]

def find_peaks_and_plot(df, show_plot=True):
    current_0_index = df['appl_current'].abs().idxmin()

    # Find positive and negative peaks
    positive_peaks, _ = find_peaks(df['dV/dI'][df['appl_current'] > 0])
    negative_peaks, _ = find_peaks(df['dV/dI'][df['appl_current'] < 0])

    # Extract peak information
    positive_peak_current = [df['appl_current'].iloc[current_0_index + i+1] for i in positive_peaks]
    positive_peak_R = [df['dV/dI'].iloc[current_0_index + i+1] for i in positive_peaks]
    negative_peak_current = df['appl_current'].iloc[negative_peaks].values
    negative_peak_R = df['dV/dI'].iloc[negative_peaks].values

    # Find maximum peaks
    positive_peak_index = np.argmax(positive_peak_R)
    negative_peak_index = np.argmax(negative_peak_R)

    if show_plot:
        # Print peak information
        print("Positive peak:")
        print(f"  Current: {positive_peak_current[positive_peak_index]}")
        print(f"  dV/dI: {positive_peak_R[positive_peak_index]:.2f}")
        print("Negative peak:")
        print(f"  Current: {negative_peak_current[negative_peak_index]}")
        print(f"  dV/dI: {negative_peak_R[negative_peak_index]:.2f}")
        # Plot the data and peaks
        plt.plot(df['appl_current'], df['dV/dI'])
        plt.scatter(positive_peak_current, positive_peak_R, color='red', label='Positive Peak')
        plt.scatter(negative_peak_current, negative_peak_R, color='blue', label='Negative Peak')
        plt.legend()
        plt.show()
    else:
        return positive_peak_current[positive_peak_index], negative_peak_current[negative_peak_index]

def get_R_fit0(run_id):
    dataset = qc.load_by_id(run_id)
    df = dataset.to_pandas_dataframe().reset_index()
    para_list = dataset.parameters.split(",")
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df['dV/dI'] = pd.DataFrame(np.diff(df['meas_voltage_K2']) / np.diff(df['appl_current']), columns=['r']).reindex(df.index, method=None)
    peaks, _ = find_peaks(df['dV/dI'])
    # Extract peak current and peak differential voltage values
    peak_current = [df['appl_current'][i] for i in peaks]
    peak_R = [df['dV/dI'][i] for i in peaks]

    # Sort the peaks based on their values
    sorted_peak_indices = np.argsort(peak_R)[::-1]  # Sort in descending order
    top_two_peak_indices = sorted_peak_indices[:2]  # Select the top two indices

    # Extract the top two peaks
    top_two_peak_current = [peak_current[i] for i in top_two_peak_indices]
    top_two_peak_R = [peak_R[i] for i in top_two_peak_indices]
    # Ensure peak0 is less than paek1
    peak0, peak1 = top_two_peak_current
    if peak1 < peak0:
        peak0, peak1 = peak1, peak0
        top_two_peak_current = peak0, peak1
    df_0 = df[df['appl_current'] < peak0]
    df_1 = df[df['appl_current'] > peak1]

    fit_0 = np.polyfit(df_0['appl_current'], df_0['meas_voltage_K2'] , 1)
    fit_1 = np.polyfit(df_1['appl_current'], df_1['meas_voltage_K2'] , 1)
    R_fit0 = fit_0[0]
    R_fit1 = fit_1[0]
    R_fit = (R_fit0 + R_fit1) / 2
    current_0_index = df['appl_current'].abs().idxmin()
    Ic = top_two_peak_current[1]-df['appl_current'].iloc[current_0_index]
    IcRn = Ic*R_fit
    return R_fit, R_fit0 ,R_fit1, top_two_peak_current, Ic, IcRn, current_0_index

def get_R_fit(run_id):
    dataset = qc.load_by_id(run_id)
    df = dataset.to_pandas_dataframe().reset_index()
    para_list = dataset.parameters.split(",")
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(
        dataset)
    df['dV/dI'] = pd.DataFrame(np.diff(df['meas_voltage_K2']) / np.diff(df['appl_current']), columns=['r']).reindex(df.index, method=None)
    Ic, Ir = find_peaks_and_plot(df, show_plot=False)
    df_0 = df[df['appl_current'] < Ir]
    df_1 = df[df['appl_current'] > Ic]
    fit_0 = np.polyfit(df_0['appl_current'], df_0['meas_voltage_K2'] , 1)
    fit_1 = np.polyfit(df_1['appl_current'], df_1['meas_voltage_K2'] , 1)
    R_fit0 = fit_0[0]
    R_fit1 = fit_1[0]
    R_fit = (R_fit0 + R_fit1) / 2
    current_0_index = df['appl_current'].abs().idxmin()
    IcRn = Ic*R_fit
    return R_fit, R_fit0 ,R_fit1, Ic, Ir, IcRn, current_0_index, df


def plot_IV_and_dVdI(run_id=1):
    dataset = qc.load_by_id(run_id)
    run_id, exp_name, sample_name, sample_junction, name, description = get_detaset_info(dataset)
    df = dataset.to_pandas_dataframe().reset_index()
    df['dV/dI'] = pd.DataFrame(np.diff(df['meas_voltage_K2']) / np.diff(df['appl_current']), columns=['r']).reindex(df.index, method=None)
    para_list = dataset.parameters.split(",")
    R_fit, R_fit0 ,R_fit1, Ic, Ir, IcRn, current_0_index, df = get_R_fit(run_id)
    # Create traces
    # Trace1 IV
    trace1 = go.Scatter(x=df['appl_current'], y=df['meas_voltage_K2'], mode='lines', name='IV')
    # Trace2 dV/dI
    trace2 = go.Scatter(x=df['appl_current'], y=df['dV/dI'], mode='lines', name='dV/dI', yaxis='y2')
    trace3 = go.Scatter(x=[Ir, Ir], y=[df['dV/dI'].min(), df['dV/dI'].max()], mode='lines', line=dict(dash='dash'), name='Peak0',yaxis='y2')
    trace4 = go.Scatter(x=[Ic, Ic], y=[df['dV/dI'].min(), df['dV/dI'].max()], mode='lines', line=dict(dash='dash'), name='Peak1',yaxis='y2')
    trace5 = go.Scatter(x=[df['appl_current'].iloc[current_0_index], df['appl_current'].iloc[current_0_index]], y=[df['dV/dI'].min(), df['dV/dI'].max()], mode='lines', line=dict(dash='dash'), name='I_0',yaxis='y2')
    trace6 = go.Scatter(x=[df['appl_current'].iloc[current_0_index], Ic], y=[0, 0], mode='lines', line=dict(dash='dash'), name='Ic')
    # Trace7 I_r
    trace7 = go.Scatter(x=[df['appl_current'].iloc[current_0_index], Ir], y=[0, 0], mode='lines', line=dict(dash='dash'), name='Ir')
    # Trace8 R_fit 
    trace8 = go.Scatter(x=[df['appl_current'].min(), df['appl_current'].max()], y=[R_fit*df['appl_current'].min(), R_fit*df['appl_current'].max()], mode='lines', line=dict(dash='dash'), name='R_fit') 
    
    # Create layout
    layout = go.Layout(
        title=f'<b>#{run_id} {exp_name} {sample_name}<br>R_fit={R_fit:.4f} Ω', 
        title_font=dict(size=25), title_x=0,
        xaxis=dict(title=para_list[0], tickfont=dict(size=16),title_font=dict(size=20)),
        yaxis=dict(title=f"{para_list[1]}", tickfont=dict(size=16),title_font=dict(size=20)),
        yaxis2=dict(title='dV/dI', tickfont=dict(size=16),title_font=dict(size=20), overlaying='y', side='right'),
        width=1000, height=1000, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h",
                    x=0.5, y=1.03,
                    xanchor="center", yanchor="top",)
    )

    fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8], layout=layout)
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=1,
                y=1,
                xref="paper",
                yref="paper",
                text=f"Ic: {Ic*1e6:.3f}µA, Ir: {Ir*1e6:.3f}µA, IcRn: {IcRn*1e6:.3f}µV",
                showarrow=False,
                font=dict(
                    size=16,
                    color="#ffffff"
                ),
                align="right",
                bgcolor="#ff7f0e",
                opacity=0.8
            )
        ]
    )
    fig.show()
    print(f"Ic = {Ic*1e6} µA\n",
          f"Ir = {Ir*1e6} µA\n", 
          f"R_fit = {R_fit} Ω\n", 
          f"IcRn = {Ic*1e6*R_fit} µV\n",
          f"R_fit0 = {R_fit0} Ω\n", 
          f"R_fit1 = {R_fit1} Ω\n",
          )
    return 

def get_R_fit_df(df):
    df = df.reset_index(drop=True)
    df['dV/dI'] = pd.DataFrame(np.diff(df['meas_voltage_K2']) / np.diff(df['appl_current']), columns=['r']).reindex(df.index, method=None)
    Ic, Ir = find_peaks_and_plot(df, show_plot=False)
    df_0 = df[df['appl_current'] < Ir]
    df_1 = df[df['appl_current'] > Ic]
    fit_0 = np.polyfit(df_0['appl_current'], df_0['meas_voltage_K2'] , 1)
    fit_1 = np.polyfit(df_1['appl_current'], df_1['meas_voltage_K2'] , 1)
    R_fit0 = fit_0[0]
    R_fit1 = fit_1[0]
    R_fit = (R_fit0 + R_fit1) / 2
    I0_idx = df['appl_current'].abs().idxmin()
    IcRn = Ic*R_fit
    return pd.Series({
        'R_fit': R_fit,
        'R_fit0': R_fit0,
        'R_fit1': R_fit1,
        'Ic': Ic,
        'Ir': Ir,
        'IcRn': IcRn,
    })

def merge_df(df):
    # 對 y_field 進行分組並應用計算函數
    results = df.groupby('y_field').apply(get_R_fit_df, include_groups=False).reset_index()

    # 將結果與原始數據合併
    df_merged = df.merge(results, on='y_field', how='left')
    return df_merged

def infer_shape_from_repeated_values(arr):
    # 初始化变量
    row_length = 0
    current_value = None
    current_count = 0
    
    # 遍历数组，寻找连续相同值的数量
    for value in arr:
        if value == current_value:
            current_count += 1
        else:
            if current_count > 0:
                if row_length == 0:
                    row_length = current_count
                elif row_length != current_count:
                    raise ValueError("数组中连续相同值的数量不一致，无法确定形状")
            current_value = value
            current_count = 1
    
    # 最后一个值的处理
    if current_count > 0:
        if row_length == 0:
            row_length = current_count
        elif row_length != current_count:
            raise ValueError("数组中连续相同值的数量不一致，无法确定形状")

    # 计算行数
    num_rows = len(arr) // row_length
    
    if len(arr) % row_length != 0:
        raise ValueError("数组长度无法整除推断出的行长度，无法确定形状")

    # 返回推断的形状
    return (num_rows, row_length)
