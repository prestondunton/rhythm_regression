from rhythm_regression import notebook_tools as nbt
from rhythm_regression import midi_processing as mp
from rhythm_regression import audio_processing as ap
from rhythm_regression import vector_processing as vp

import librosa
import matplotlib.pyplot as plt
import mido
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from rhythm_regression.unit_conversion import MILLISECONDS_PER_SECOND, SECONDS_PER_MINUTE

TEMP_DIRECTORY = '.temp'
AUDIO_COLORS = ['#215097', '#F9C80E', '#ED6A5A', '#7297AC', '#1cbd5c']
MIDI_COLOR = 'black'
METRIC_ROUND_PLACES = 2


def render_app():

    st.set_page_config(layout="wide")
    st.title('Rhythm Regression')

    audio_files = render_file_loaders()

    center_mode = st.sidebar.selectbox('Centering Mode', ['Error Mean 0', 'First Note'])

    if len(audio_files) > 0:
        render_audio_playbacks(audio_files)

    if 'midi' in st.session_state and 'audios' in st.session_state:
        compute_transient_midi_vectors(center_mode)
        compute_error_vectors()
        compute_tempo_vectors()
        compute_summary_stats()
        
        set_default_time_range()

        render_dot_plots()

        st.subheader('Tempo')
        col1, col2 = st.columns([1,6])
        with col1:
            render_tempo_metrics()
        with col2:
            render_tempo_plot()

        st.subheader('Errors')
        col1, col2 = st.columns([1,7])
        with col1:
            render_error_metrics()
        with col2:
            render_error_plot()

        render_summary_table()


def render_file_loaders():
    st.sidebar.header('Upload Files')
    midi_file = st.sidebar.file_uploader('Upload a MIDI File', type=['mid', 'midi'])
    audio_files = st.sidebar.file_uploader('Upload an Audio File', type=['mp3', 'm4a', 'wav'],
                                        accept_multiple_files=True)
    audio_files.sort(key= lambda file: file.name)
    
    if midi_file is not None:
        load_midi(midi_file)

    if len(audio_files) > 0:
        st.session_state['audio_names'] = [file.name for file in audio_files]
        st.session_state['audios'] = [load_audio(file) for file in audio_files]
        st.session_state['num_audios'] = len(audio_files)

    return audio_files

        
def load_midi(midi_file):
    st.session_state['midi_name'] = midi_file.name
    temp_midi_path = os.path.join(TEMP_DIRECTORY, midi_file.name)

    with open(temp_midi_path, 'wb') as f: 
        f.write(midi_file.getbuffer())         

    st.session_state['midi'] = mido.MidiFile(temp_midi_path)
    os.remove(temp_midi_path)


@st.experimental_memo(show_spinner=False)
def load_audio(file):
    with st.spinner(f'Loading {file.name}'):
        temp_audio_path = os.path.join(TEMP_DIRECTORY, file.name)

        with open(temp_audio_path, 'wb') as f: 
            f.write(file.getbuffer())         

        audio, sampling_rate = librosa.load(temp_audio_path)

        os.remove(temp_audio_path)
        return audio


def render_audio_playbacks(audio_files):
    for i in range(len(audio_files)):
        st.subheader(st.session_state['audio_names'][i])
        st.audio(audio_files[i])


def compute_transient_midi_vectors(center_mode):
    transient_vectors = [ap.rms_energy_transients(audio) for audio in st.session_state['audios']]
    midi_vector = mp.get_midi_vector(st.session_state['midi'])
    bpm = mp.get_bpm(st.session_state['midi'])

    for i in range(len(transient_vectors)):
            print(f'{st.session_state["audio_names"][i]} has {len(transient_vectors[i])} transients')

    for i in range(len(transient_vectors)):
        if len(transient_vectors[i]) < len(midi_vector):
            transient_vectors[i] = vp.add_nan_transients(transient_vectors[i], midi_vector)
        elif len(transient_vectors[i]) > len(midi_vector):
            transient_vectors[i] = vp.delete_transients(transient_vectors[i], midi_vector)
    
    for transient_vector in transient_vectors:
        if center_mode == 'First Note':
            transient_vector -= transient_vector.min()
        else:
            vp.center_transients_on_midi(transient_vector, midi_vector)

    st.session_state['transient_vectors'] = transient_vectors
    st.session_state['midi_vector'] = midi_vector
    st.session_state['bpm'] = bpm


def compute_error_vectors():

    error_vectors = [st.session_state['midi_vector'] - transient_vector
                                        for transient_vector in st.session_state['transient_vectors']]

    st.session_state['error_vectors'] = error_vectors

    st.session_state['mean_error_vector'] =  np.mean(np.array(error_vectors), axis=0)


def compute_tempo_vectors():

    tempo_vectors = [vp.get_tempo_vector(transient_vector, st.session_state['midi_vector'], st.session_state['bpm']) 
                    for transient_vector in st.session_state['transient_vectors']]

    st.session_state['tempo_vectors'] = tempo_vectors

    st.session_state['mean_tempo_vector'] =  np.mean(np.array(tempo_vectors), axis=0)


def compute_summary_stats():

    average_errors = [np.nanmean(np.abs(error_vector)) for error_vector in st.session_state['error_vectors']]

    total_errors = [np.nansum(np.abs(error_vector)) for error_vector in st.session_state['error_vectors']]

    error_deviations = [np.nanstd(error_vector) for error_vector in st.session_state['error_vectors']]

    average_tempos = [np.nanmean(tempo_vector) for tempo_vector in st.session_state['tempo_vectors']]

    tempo_deviations = [np.nanstd(tempo_vector) for tempo_vector in st.session_state['tempo_vectors']]

    summary_stats_df = pd.DataFrame({'Average Error (s)': average_errors, 'Total Error (s)': total_errors,
                                     'Error Deviation (s)': error_deviations, 'Average Tempo (bpm)': average_tempos,
                                     'Tempo Deviation (bpm)': tempo_deviations}, index=st.session_state['audio_names'])

    st.session_state['summary_stats_df'] = summary_stats_df


def set_default_time_range():
    minimum = st.session_state['midi_vector'].min().item()
    maximum = st.session_state['midi_vector'].max().item()

    for transient_vector in st.session_state['transient_vectors']:
        minimum = min(minimum, np.nanmin(transient_vector).item())
        maximum = max(maximum, np.nanmax(transient_vector).item())

    st.session_state['default_time_range'] = (minimum, maximum)


def render_dot_plots():

    time_range = st.slider('Time Range', value=st.session_state['default_time_range'])

    for i in range(len(st.session_state['transient_vectors'])):

        transient_vector = st.session_state['transient_vectors'][i]
        midi_vector = st.session_state['midi_vector']
        bpm = st.session_state['bpm']

        axs = nbt.plot_midi_vector(midi_vector, bpm, time_range=time_range, 
                                   units='s', y_level=0.333, figsize=(20,1), color=MIDI_COLOR)
        nbt.plot_midi_vector(transient_vector, bpm, time_range=time_range, units='s', y_level=0.666, 
                            axs=axs, color=AUDIO_COLORS[i%len(AUDIO_COLORS)], 
                            title=st.session_state['midi_name'] + ' / ' + st.session_state['audio_names'][i])
        st.pyplot(plt.gcf())

    
    num_rows = 1 + len(st.session_state['transient_vectors'])
    if num_rows > 2:
        axs = nbt.plot_midi_vector(midi_vector, bpm, time_range=time_range, 
                                units='s', y_level=1 / (num_rows+1), figsize=(20,1), color=MIDI_COLOR)
        for i in range(len(st.session_state['transient_vectors'])):
            nbt.plot_midi_vector(st.session_state['transient_vectors'][i], bpm, time_range=time_range, units='s', 
                                y_level= (i+2) / (num_rows+1), 
                            axs=axs, color=AUDIO_COLORS[i%len(AUDIO_COLORS)])
        plt.title(st.session_state['midi_name'] + ' / ' + ' / '.join(st.session_state['audio_names']))
        st.pyplot(plt.gcf())


def render_tempo_metrics():

    st.markdown('#')
    st.markdown('#')

    stats_df = st.session_state['summary_stats_df']

    most_recent_sample = st.session_state['audio_names'][-1]

    average_tempo = round(stats_df.loc[most_recent_sample]['Average Tempo (bpm)'], METRIC_ROUND_PLACES)
    tempo_deviation = round(stats_df.loc[most_recent_sample]['Tempo Deviation (bpm)'], METRIC_ROUND_PLACES)

    if st.session_state['num_audios'] > 1:
        sample_before_that = st.session_state['audio_names'][-2]

        delta_average_tempo = round(average_tempo - stats_df.loc[sample_before_that]['Average Tempo (bpm)'], METRIC_ROUND_PLACES)
        delta_tempo_deviation = round(tempo_deviation - stats_df.loc[sample_before_that]['Tempo Deviation (bpm)'], METRIC_ROUND_PLACES)
    
    else:
        delta_average_tempo = None
        delta_tempo_deviation = None
    
    st.subheader(most_recent_sample)
    average_tempo_delta_color = 'normal' if average_tempo < st.session_state['bpm'] else 'inverse'
    st.metric('Average Tempo', f'{average_tempo} bpm', delta=delta_average_tempo, delta_color=average_tempo_delta_color)
    st.metric('Tempo Deviation', f'{tempo_deviation} bpm', delta=delta_tempo_deviation, delta_color='inverse')


def render_tempo_plot():

    tempo_vectors = st.session_state['tempo_vectors']
    vector_names = st.session_state['audio_names']

    if len(tempo_vectors) > 1:
        tempo_vectors = tempo_vectors + [st.session_state['mean_tempo_vector']]
        vector_names = vector_names + ['Mean']
    
    num_tempo_vectors = len(tempo_vectors)
    points_per_vector = len(st.session_state['midi_vector']) - 1
    
    plot_df = pd.DataFrame({'Time (s)': np.tile(st.session_state['midi_vector'][:-1], num_tempo_vectors),
                            'Tempo (bpm)': np.concatenate(tempo_vectors),
                            'color': np.repeat(vector_names, points_per_vector)
                            })
    
    fig = px.line(plot_df, x='Time (s)', y='Tempo (bpm)', color='color', color_discrete_sequence=AUDIO_COLORS, markers=True)
    st.plotly_chart(fig, use_container_width=True)


def render_error_metrics():

    st.markdown('#')
    
    stats_df = st.session_state['summary_stats_df']

    most_recent_sample = st.session_state['audio_names'][-1]

    average_error = round(stats_df.loc[most_recent_sample]['Average Error (s)'], METRIC_ROUND_PLACES)
    total_error = round(stats_df.loc[most_recent_sample]['Total Error (s)'], METRIC_ROUND_PLACES)
    error_deviation = round(stats_df.loc[most_recent_sample]['Error Deviation (s)'], METRIC_ROUND_PLACES)

    if st.session_state['num_audios'] > 1:
        sample_before_that = st.session_state['audio_names'][-2]

        delta_average_error = round(average_error - stats_df.loc[sample_before_that]['Average Error (s)'], METRIC_ROUND_PLACES)
        delta_total_error = round(total_error - stats_df.loc[sample_before_that]['Total Error (s)'], METRIC_ROUND_PLACES)
        delta_error_deviation = round(error_deviation - stats_df.loc[sample_before_that]['Error Deviation (s)'], METRIC_ROUND_PLACES)
        
    else:
        delta_average_error = None
        delta_total_error = None
        delta_error_deviation = None
    
    st.subheader(most_recent_sample)
    st.metric('Average Error', f'{average_error} s', delta=delta_average_error, delta_color='inverse')
    st.metric('Total Error', f'{total_error} s', delta=delta_total_error, delta_color='inverse')
    st.metric('Error Deviation', f'{error_deviation} s', delta=delta_error_deviation, delta_color='inverse')


def render_error_plot():

    error_vectors = st.session_state['error_vectors']
    vector_names = st.session_state['audio_names']

    if len(error_vectors) > 1:
        error_vectors = error_vectors + [st.session_state['mean_error_vector']]
        vector_names = vector_names + ['Mean']
    
    num_error_vectors = len(error_vectors)
    points_per_vector = len(st.session_state['midi_vector'])
    
    plot_df = pd.DataFrame({f'Time (s)': np.tile(st.session_state['midi_vector'], num_error_vectors),
                            f'Error (s)': np.concatenate(error_vectors),
                            'color': np.repeat(vector_names, points_per_vector)
                            })
    
    fig = px.line(plot_df, x=f'Time (s)', y=f'Error (s)', color='color', color_discrete_sequence=AUDIO_COLORS, markers=True)
    st.plotly_chart(fig, use_container_width=True)


def render_summary_table():
    st.header('Metrics')

    stats_df = st.session_state['summary_stats_df']

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        st.dataframe(stats_df)
    with col2:
        chart_column = st.selectbox('Pick a column to plot', stats_df.columns)
        fig = px.bar(stats_df, x=stats_df.index, y=chart_column, color_discrete_sequence=['black'])
        st.plotly_chart(fig, use_container_width=True)
    


if __name__ == '__main__':
    render_app()