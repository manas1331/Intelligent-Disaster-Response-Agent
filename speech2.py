import pyaudio
import wave
from pynput import keyboard

# Recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"

frames = []
stop_recording = False

def on_press(key):
    global stop_recording
    try:
        if key.char == 'q':
            stop_recording = True
            return False  # Stop the listener
    except AttributeError:
        pass

def record():
    global frames
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording... Press 'q' to stop.")

    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Saved recording to {WAVE_OUTPUT_FILENAME}")

# Start keyboard listener in a thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start recording in main thread
record()
