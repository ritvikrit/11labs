import os
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from playsound  import playsound
import time

# -------------------------
# Load API Keys
# -------------------------
load_dotenv()

# -------------------------
# Initialize Clients
# -------------------------
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

eleven_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

# -------------------------
# Record Audio
# -------------------------
duration = 5  # seconds
sample_rate = 16000  # Hz
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful voice assistant. Reply naturally and briefly."
    }
]
while True:
    input("Press ENTER to start recording...")

    print("Recording... Speak now.")

    recording = sd.rec(
        999999,
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )

    input("Press ENTER to stop recording...")

    sd.stop()

    print("Recording complete.")

    # Save recording
    write("input.wav", sample_rate, recording)

    # -------------------------
    # Speech To Text
    # -------------------------
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=open("input.wav", "rb"),
        language="en",
        prompt="The conversation is about AI, GPT models, and technology."
    )

    user_text = transcript.text

    print("\nYou said:")
    print(user_text)

    # -------------------------
    # GPT Reply
    # -------------------------
    conversation_history.append(
        {
            "role": "user",
            "content": user_text
                }
        )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history
    )

    ai_reply = response.choices[0].message.content
    conversation_history.append(
        {
            "role": "assistant",
            "content": ai_reply
        }
    )

    print("\nAI replied:")
    print(ai_reply)

    # -------------------------
    # Text To Speech
    # -------------------------
    audio = eleven_client.text_to_speech.convert(
        text=ai_reply,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2"
    )

    output_path = f"reply_{int(time.time())}.mp3"

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    playsound(output_path)

