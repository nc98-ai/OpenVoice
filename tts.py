import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import time

start =time.time()

if torch.cuda.is_available() :
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("Pas de GPU dispo -> utilisation du CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device utilisé: " + str(device))


exit()

ckpt_converter = 'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
print (f"device:{device} - {(time.time() -start):.4}s")

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
print(f"***** outputdir créé - {(time.time() -start):.4}s")


# definition du ton donné au texte
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
print(f"***** definition du ton- {(time.time() -start):.4}s")


# 
from openai import OpenAI
from dotenv import load_dotenv

# Please create a file named .env and place your
# OpenAI key as OPENAI_API_KEY=xxx
load_dotenv() 
api_key=os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print(f"OPENAI_API_KEY: {api_key} ")

model = "tts-1" #tts-1-hd
voice = "nova"
response = client.audio.speech.create(
    model=model,
    voice=voice,
    input="This audio will be used to extract the base speaker tone color embedding. " + \
        "Typically a very short audio should be sufficient, but increasing the audio " + \
        "length will also improve the output audio quality."
)
print(f"model: {model} ; voice:{voice}; ")

response.stream_to_file(f"{output_dir}/openai_source_output_{model}_{voice}.mp3")

print(f"***** génération de la sortie avec OpenAI dans {output_dir}/openai_source_output_{model}_{voice}.mp3 -  {(time.time() -start):.4}s")

# base_speaker = f"{output_dir}/openai_source_output.mp3"
# source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)

# reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
# target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# print(f"***** voice to use  -  {(time.time() -start):.4}s")


# text = [
#     "MyShell est une plateforme décentralisée et complète pour découvrir, créer et miser sur des applications natives d'IA."
# ]



# voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
# for voice in voices:
#     for i, t in enumerate(text):

#         model = "tts-1"
        
#         response = client.audio.speech.create(
#             model=model,
#             voice=voice,
#             input=t,
#         )
#         src_path = f'{output_dir}/tmp_{model}_{voice}.wav'

#         response.stream_to_file(src_path)

#         save_path = f'{output_dir}/output_crosslingual-{voice} _{i}.wav'

#         # Run the tone color converter
#         encode_message = "@MyShell"
#         tone_color_converter.convert(
#             audio_src_path=src_path, 
#             src_se=source_se, 
#             tgt_se=target_se, 
#             output_path=save_path,
#             message=encode_message)

        
#         print(f"***** génération de la sortie avec OpenAI dans {output_dir}/openai_source_output_{model}_{voice}.mp3 -  {(time.time() -start):.4}s")