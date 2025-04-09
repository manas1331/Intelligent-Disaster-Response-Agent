# UI for Intelligent Disaster Response Chatbot

import streamlit as st
from streamlit_mic_recorder import mic_recorder
import random
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
import io
import os

AudioSegment.converter = which("ffmpeg") or r"C:\Users\91984\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

st.set_page_config(page_title="Disaster Chatbot", layout="wide")

# Session states
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: #1f77b4;'>Disaster-Aware Chatbot</h1>
        <p style='font-size: 1.2rem;'>Empowering Relief Through Conversations.</p>
    </div>
""", unsafe_allow_html=True)

# Mode toggle button
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🌍 Switch Chat Mode", use_container_width=True):
        st.session_state.mode = "disaster" if st.session_state.mode == "chat" else "chat"

# Initialize input
user_input = ""
text_input = ""

# Main Chat Mode
if st.session_state.mode == "chat":
    st.markdown("### 🤖 General Chat Mode")

    # Microphone Input
    with st.expander("🎙️ Speak Instead of Typing", expanded=True):
        audio = mic_recorder(start_prompt="Start Talking", stop_prompt="Stop Recording", key='recorder')
        if audio:
            try:
                st.info("⏳ Transcribing your voice...")
                sound = AudioSegment.from_file(io.BytesIO(audio["bytes"]), format="wav")
                sound.export("temp.wav", format="wav")

                r = sr.Recognizer()
                with sr.AudioFile("temp.wav") as source:
                    audio_data = r.record(source)
                    user_input = r.recognize_google(audio_data)

                os.remove("temp.wav")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    # Text Input
    text_input = st.text_input("💬 Type your message:", placeholder="Ask something...", key="text_prompt")

    # Use spoken input if available, else text input
    final_input = user_input or text_input

    if final_input:
        st.session_state.chat_history.append(("user", final_input))
        bot_response = f"I'm still learning. You said: '{final_input}'"
        st.session_state.chat_history.append(("bot", bot_response))

    # Display chat history
    st.markdown("### Chat History")
    chat_pairs = list(zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]))[::-1]

    for user_msg, bot_msg in chat_pairs:
        if user_msg[0] == "user":
            st.markdown(f"<div style='background-color: black; padding: 10px; border-radius: 8px;'> <strong>You:</strong> {user_msg[1]} </div>", unsafe_allow_html=True)
        if bot_msg[0] == "bot":
            st.markdown(f"<div style='background-color: grey; padding: 10px; border-radius: 8px;'> <strong>Bot:</strong> {bot_msg[1]} </div>", unsafe_allow_html=True)

# Disaster Info Mode
else:
    st.markdown("### 🌍 Disaster Info Chat")

    # Disaster dropdown
    st.markdown("Pick a recent disaster to learn more:")
    disasters = ["🔥 California Wildfires", "🌊 Mumbai Floods", "🌪️ Oklahoma Tornadoes", "🌍 Syria Earthquake"]
    selected_disaster = st.selectbox("Choose a disaster", disasters)

    if selected_disaster:
        st.markdown(f"**You selected:** {selected_disaster}")
        st.markdown(f"**Bot:** Fetching real-time data for **{selected_disaster}**... [Mock response here]")

# Footer
st.markdown("""
    <hr style='border-top: 1px solid #ccc;' />
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        All rights reserved.
    </div>
""", unsafe_allow_html=True)
