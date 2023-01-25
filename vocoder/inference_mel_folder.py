import os
import json
import numpy as np
from glob import glob

import torch
from scipy.io.wavfile import write

from .models import Generator
from .denoiser import Denoiser


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_vocoder(vocoder_path, config_path, to_cuda=False):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'blur' in vocoder_path:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.5
    else:
        if 'gaussian_blur' in config_vocoder:
            config_vocoder['gaussian_blur']['p_blurring'] = 0.0
        else:
            config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
            h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    vocoder.eval()
    denoiser.eval()

    return vocoder, denoiser


def inference(input_mel_folder, vocoder_path, vocoder_config_path, denoising_strength):
    vocoder, denoiser = load_vocoder(vocoder_path, vocoder_config_path)

    with torch.no_grad():
        files_all = []
        for input_mel_file in glob(input_mel_folder +'/*.mel'):
            x = torch.load(input_mel_file)
            audio = vocoder(x).float()[0]
            audio_denoised = denoiser(
                audio, strength=denoising_strength)[0].float()

            audio = audio[0].cpu().numpy()
            audio_denoised = audio_denoised[0].cpu().numpy()
            audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))
            audio_denoised = audio_denoised.astype('int16')

            output_file = input_mel_file.replace('.mel','.wav')
            write(output_file, 22050, audio_denoised)
            print('<<--',output_file)

            files_all.append(output_file)

            os.remove(input_mel_file)

        names = []
        for k in files_all:
            names.append(int(k.replace(input_mel_folder,'').replace('/','').replace('.wav','')))

        names_w = [f'{it}.wav' for it in sorted(names)]

        print('sox ' + ' '.join(names_w) + ' all.wav')


def process_folder(input_mel_folder, vocoder_path, vocoder_config_path, denoising_strength):
    inference(input_mel_folder, vocoder_path, vocoder_config_path, denoising_strength)
