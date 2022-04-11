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
import re
import streamlit as st

from rhythm_regression.unit_conversion import MILLISECONDS_PER_SECOND

TEMP_DIRECTORY = '.temp'
TOY_DATA_DIR = 'toy_data'
AUDIO_COLORS = ['#D74E09', '#3F88C5', '#F2BB05', '#0B6E4F', '#63CCCA', 'black']
MIDI_COLOR = 'black'
METRIC_ROUND_PLACES = 2


def render_app():

    init_app()

    render_sidebar()
    st.title('Rhythm Regression™')
    st.subheader('By Preston Dunton')
    render_introduction()
    render_how_to()

    if st.session_state['num_audios'] > 0:
        render_audio_playbacks(st.session_state['audio_files'])

    if 'midi' in st.session_state and st.session_state['num_audios'] > 0:
        compute_transient_midi_vectors()
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

        if st.session_state['num_audios'] > 1:
            render_metrics_section()

        render_error_histogram()


def init_app():
    st.set_page_config(page_title='Rhythm Regression', page_icon='./images/favicons/favicon.ico', layout='wide')

    if 'num_audios' not in st.session_state:
       st.session_state['num_audios'] = 0 
    
    if not os.path.exists(TEMP_DIRECTORY):
        os.mkdir(TEMP_DIRECTORY)


def render_sidebar():
    
    st.sidebar.header('Upload Files')
    render_midi_loading()
    st.sidebar.markdown('#')
    st.sidebar.markdown('#')
    render_audio_loading()

    st.sidebar.markdown('#')
    st.sidebar.markdown('#')
    st.sidebar.selectbox('Error Centering Mode', ['Equal Error', 'First Note'], key='center_mode')
    st.sidebar.write('The error centering mode is used for calculating error.  "Equal Error" assumes '
                     'that the player is equally likely to be ahead / behind, wheras "First Note" '
                     'assumes the first note is always in time.')


def render_introduction():

    with st.expander('Show / Hide Introduction', expanded=True):

        st.write("Among the different styles of drummers, marching drummers have a reputation for an "
                "extremely high standard for precision and consistency.  For example, some marching "
                "drummers care about timing differences even down to the resolution of milliseconds.  In order "
                "to achieve these standards, some drummers log and monitor their practice by recording "
                "maximum tempos achieved and by recording themselves playing.  Slow motion cameras in newer "
                "smartphones used to watch stick movements are an example of a tool recently adopted by drummers."
                " Methods like these can be effective, but are not automated, and are most often qualitative. "
                "Qualitative methods are problematic because inexperienced players usually have poor judgment on "
                "their own skills, and how to improve them. ")

        st.write("This capstone project is my thesis for the Honors Program here at CSU.  One major theme in this "
                "thesis is to bridge the gap between qualitative methods for evaluation and the high standards for "
                "precision and consistency.  It is hypothesized that quantitative methods and AI tools can provide "
                "players insight into their playing that goes beyond a player’s level of precise listening.  "
                "Quantitative methods, along with computer science, can also provide drummers a way to automatically "
                "log their practice sessions and monitor progress on music. The goals of this project are to explore"
                " marching percussion audio data in depth, provide effective summarization of the data, and provide "
                "users a way to receive automated feedback about their playing.")

        st.write("If you're interested in this project, and would like learn how it works, I'll be giving a presentation"
                " in the Computer Science Building CSB 130 on Friday, April 29th from 1-2pm.  You can also contact "
                "me at preston.dunton@gmail.com or just text me if you have my number.")


def render_how_to():
    with st.expander('Show / Hide How To', expanded=True):
        st.write("This web app allows you to upload MIDI files of music, and audio recordings of you playing it."
                 "In the sidebar, you can either upload your own MIDI and audio recordings or use some that I have"
                 "provided.  You can only use one MIDI file at a time, but you can use several audio files.  "
                 "This allows you to compare different takes of you playing something.")

        st.write("Record audio in a quiet room, and do not play any extra notes, or make any unintended sounds."
                 "  The audio processing used in this project isn't perfect, so it helps if you're recording is "
                "as clean as possible.  That being said, you do not need a fancy microphone.  I used my iPhone "
                "11 microphone for this project.  If you notice any errors reported that are very large (>0.2s), you may "
                "assume that the audio processing made a mistake.  State-of-the-art audio processing and live "
                "recording aren't currently supported, as this project is in its (very) early stages of development.")

        st.write("If you encounter a code error marked by a red box with code inside of it, please send me a "
                 "screenshot, and the data you were using.  This will help me significantly.  Also try "
                 "refreshing the page and reuploading your files.  Let me know if you have other problems.")


def render_midi_loading():
    
    midi_option = st.sidebar.selectbox('Choose a MIDI File',
                                        options=sorted(os.listdir(os.path.join(TOY_DATA_DIR, 'midi'))+['Upload your own']))
    if midi_option == 'Upload your own':
        midi_file = st.sidebar.file_uploader('Upload a MIDI File', type=['mid', 'midi'])
        if midi_file is not None:
            load_midi(midi_file)
    else:
        sheets_filename = re.sub('_\d{3}bpm.mid', '.pdf', midi_option)
        with open(os.path.join(TOY_DATA_DIR, 'sheets', sheets_filename), 'rb') as sheets_file:    
            st.sidebar.download_button('Download Sheet Music for this MIDI', 
                                    data=sheets_file, file_name=sheets_filename, mime='application/pdf')

        midi_path = os.path.join(TOY_DATA_DIR, 'midi', midi_option)
        st.session_state['midi_name'] = midi_option
        st.session_state['midi'] = mido.MidiFile(midi_path)


def load_midi(midi_file):
    
    temp_midi_path = os.path.join(TEMP_DIRECTORY, midi_file.name)
    with open(temp_midi_path, 'wb') as f: 
        f.write(midi_file.getbuffer())         

    st.session_state['midi_name'] = midi_file.name
    st.session_state['midi'] = mido.MidiFile(temp_midi_path)

    os.remove(temp_midi_path)


def render_audio_loading():

    audio_options = st.sidebar.multiselect('Choose (an) Audio File(s)',
                                        options=sorted(os.listdir(os.path.join(TOY_DATA_DIR, 'audio'))+['Upload your own']))

    audio_names = [option for option in audio_options if option != 'Upload your own']
    audios = [load_toy_audio(audio_name) for audio_name in audio_names]
    audio_files = [os.path.join(TOY_DATA_DIR, 'audio', audio_name) for audio_name in audio_names]

    if 'Upload your own' in audio_options:
        uploaded_audio_files = st.sidebar.file_uploader('Upload an Audio File', type=['mp3', 'm4a', 'wav'],
                                            accept_multiple_files=True)
        audio_names += [file.name for file in uploaded_audio_files]
        audio_files += uploaded_audio_files
        audios += [load_audio(file) for file in uploaded_audio_files]
        
    st.session_state['num_audios'] = len(audios)
    #audio_files.sort(key= lambda file: file.name)
    if len(audios) > 0:
        st.session_state['audio_files'] = audio_files
        st.session_state['audio_names'] = audio_names
        st.session_state['audios'] = audios
            

@st.experimental_memo(show_spinner=False)
def load_toy_audio(toy_audio_name):
    with st.spinner(f'Loading {toy_audio_name}'):
        audio_path = os.path.join(TOY_DATA_DIR, 'audio', toy_audio_name)
        audio, sampling_rate = librosa.load(audio_path)
        return audio

        
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


def compute_transient_midi_vectors():
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
        if st.session_state['center_mode'] == 'First Note':
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

    average_errors = [np.nanmean(np.abs(error_vector)) * MILLISECONDS_PER_SECOND for error_vector in st.session_state['error_vectors']]

    total_errors = [np.nansum(np.abs(error_vector)) for error_vector in st.session_state['error_vectors']]

    error_deviations = [np.nanstd(np.abs(error_vector))  * MILLISECONDS_PER_SECOND for error_vector in st.session_state['error_vectors']]

    average_tempos = [np.nanmean(tempo_vector) for tempo_vector in st.session_state['tempo_vectors']]

    tempo_deviations = [np.nanstd(tempo_vector) for tempo_vector in st.session_state['tempo_vectors']]

    summary_stats_df = pd.DataFrame({'Average Error (ms)': average_errors, 'Total Error (s)': total_errors,
                                     'Error Deviation (ms)': error_deviations, 'Average Tempo (bpm)': average_tempos,
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
    st.write('Note: The black dots in the plots below are where the notes should occur according to MIDI.  '
             'The lines are beats.')

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

    average_error = round(stats_df.loc[most_recent_sample]['Average Error (ms)'], METRIC_ROUND_PLACES)
    total_error = round(stats_df.loc[most_recent_sample]['Total Error (s)'], METRIC_ROUND_PLACES)
    error_deviation = round(stats_df.loc[most_recent_sample]['Error Deviation (ms)'], METRIC_ROUND_PLACES)

    if st.session_state['num_audios'] > 1:
        sample_before_that = st.session_state['audio_names'][-2]

        delta_average_error = round(average_error - stats_df.loc[sample_before_that]['Average Error (ms)'], METRIC_ROUND_PLACES)
        delta_total_error = round(total_error - stats_df.loc[sample_before_that]['Total Error (s)'], METRIC_ROUND_PLACES)
        delta_error_deviation = round(error_deviation - stats_df.loc[sample_before_that]['Error Deviation (ms)'], METRIC_ROUND_PLACES)
        
    else:
        delta_average_error = None
        delta_total_error = None
        delta_error_deviation = None
    
    st.subheader(most_recent_sample)
    st.metric('Average Error', f'{average_error} ms', delta=delta_average_error, delta_color='inverse')
    st.metric('Total Error', f'{total_error} s', delta=delta_total_error, delta_color='inverse')
    st.metric('Error Deviation', f'{error_deviation} ms', delta=delta_error_deviation, delta_color='inverse')


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


def render_metrics_section():
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
        fig = px.bar(stats_df, x=stats_df.index, y=chart_column, color=stats_df.index, color_discrete_sequence=AUDIO_COLORS, labels={
                     "index": "Sample"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_error_histogram():
    st.header('Error Histogram')
    points_per_audio = len(st.session_state['error_vectors'][0])
    plot_df = pd.DataFrame({'Sample': np.tile(st.session_state['audio_names'], points_per_audio),
                            'Errors (s)': np.concatenate(st.session_state['error_vectors'])
                            })
    fig = px.histogram(plot_df, x='Errors (s)', color='Sample', color_discrete_sequence=AUDIO_COLORS)
    st.plotly_chart(fig)
    

if __name__ == '__main__':
    render_app()