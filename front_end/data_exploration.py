import librosa
import streamlit as st

print('Loading Audio')
audio, sampling_rate = librosa.load('./data/audio/Sample 95.m4a')

st.audio('./data/audio/Sample 95.m4a')
st.line_chart(audio)
