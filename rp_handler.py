from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import tempfile
import os
import base64
import numpy as np
import scipy.io.wavfile
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier
import librosa
import runpod



# Load models globally for reuse across invocations
MODEL_ID = "Lingua-Connect/speecht5_sw_bible"
processor = SpeechT5Processor.from_pretrained(MODEL_ID)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
dataset_text_column = "words"
dataset_audio_column = "audio"

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
vocoder.to(device)



# create speaker embeddings function

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

# Load speaker embeddings from a dataset for voice cloning in this case its the speaker from the dataset we used to train
embeddings_dataset = load_dataset("kazeric/OpenBible_Swahili_book_split", "HAB_clean",split="train", streaming=True)
example = next(iter(embeddings_dataset))

#get the info from the data item
original_audio =example[dataset_audio_column]['array']
original_sr = example[dataset_audio_column]['sampling_rate']
target_sr = 16000

#resample the audio to 16K as required by the model
example[dataset_audio_column]['array'] = librosa.resample(original_audio, orig_sr=original_sr, target_sr=target_sr)
example[dataset_audio_column]['sampling_rate'] = target_sr

# shape the embeddings to the rigth shape [1,512]
speaker_embeddings = create_speaker_embedding(example[dataset_audio_column]["array"])
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
speaker_embeddings = speaker_embeddings.to(device)


def handler(event):
    try:
        # Get text from the request
        if "input" not in event or "text" not in event["input"]:
            return {"error": "No text provided"}

        text = event["input"]["text"]

        # Prepare text input
        inputs = processor(text=text, return_tensors="pt")

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate speech
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # Convert speech tensor to numpy array if needed
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()

        # Save the speech to a temporary file
        output_path = os.path.join(tempfile.gettempdir(), "output.wav")
        scipy.io.wavfile.write(output_path, rate=16000, data=speech)

        # Read the file and encode to base64
        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Clean up
        os.unlink(output_path)

        return {
            "success": True,
            "audio_data": audio_base64,
            "sample_rate": 16000,
            "format": "wav"
        }

    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})