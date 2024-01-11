import torch
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment
from datasets import load_dataset
import numpy as np

tokenizer=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model=Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 
 # tokenize
inputs= tokenizer(ds[9  ]["audio"]["array"], sampling_rate=16000,return_tensors="pt", padding="longest").input_values

r=sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print("Speak now..")
    while True:
        audio=r.listen(source) #pyaudio object
        #print("Heard")
        data=io.BytesIO(audio.get_wav_data()) #list of bytes
        clip=AudioSegment.from_file(data)   #numpy array
        x=torch.FloatTensor(clip.get_array_of_samples()) #tensor

        inputs=tokenizer(x,sampling_rate=16000,return_tensors='pt',padding='longest').input_values
        logits=model(inputs).logits
        tokens=torch.argmax(logits,axis=-1)
        text=tokenizer.batch_decode(tokens)

        print("You said: ",str(text).lower())

"""
logits=model(inputs).logits
tokens=torch.argmax(logits,axis=-1)
text=tokenizer.batch_decode(tokens)
print("You said: ",str(text)
"""      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      )