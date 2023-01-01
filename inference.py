import argparse
import os
import json

import torch
from torch.cuda import amp

from radtts import RADTTS
from data import Data
from common import update_params
from vocoder.inference_mel_folder import process_folder


def lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    return [f.rstrip() for f in files]


def infer(radtts_path, vocoder_path, vocoder_config_path, text_path, speaker,
          speaker_text, speaker_attributes, sigma, sigma_tkndur, sigma_f0,
          sigma_energy, f0_mean, f0_std, energy_mean, energy_std,
          token_dur_scaling, denoising_strength, n_takes, output_dir, use_amp,
          plot, seed):

    torch.manual_seed(seed)

    radtts = RADTTS(**model_config)
    radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs

    weights = torch.load(radtts_path, map_location='cpu')
    radtts.load_state_dict(weights, strict=False)
    radtts.eval()

    print("Loaded checkpoint '{}')" .format(radtts_path))

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    speaker_id = torch.LongTensor([0])
    speaker_id_text = torch.LongTensor([0])
    speaker_id_attributes = 'lada'
    speaker_id_attributes = torch.LongTensor([0])

    text_list = lines_to_list(text_path)

    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(text_list):
        if text.startswith("#"):
            continue
        print("{}/{}: {}".format(i, len(text_list), text))
        text = trainset.get_text(text)[None]
        for take in range(n_takes):
            with amp.autocast(use_amp):
                with torch.no_grad():
                    outputs = radtts.infer(
                        speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                        sigma_energy, token_dur_scaling, token_duration_max=100,
                        speaker_id_text=speaker_id_text,
                        speaker_id_attributes=speaker_id_attributes,
                        f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                        energy_std=energy_std)

                    mel = outputs['mel']
                    filename_mel = f'{output_dir}/{i}.mel'

                    torch.save(mel, filename_mel)

    # Use vocoder to convert MELs to WAVs    
    process_folder(output_dir, vocoder_path, vocoder_config_path, denoising_strength)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file config')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-r', '--radtts_path', type=str)
    parser.add_argument('-t', '--text_path', type=str)
    parser.add_argument('-vcf', '--vocoder_path', type=str)
    parser.add_argument('-vcp', '--vocoder_config_path', type=str)
    parser.add_argument('-d', '--denoising_strength', type=float, default=0.0)
    parser.add_argument('-o', "--output_dir", default="results")
    parser.add_argument("--sigma", default=0.8, type=float, help="sampling sigma for decoder")
    parser.add_argument("--sigma_tkndur", default=0.666, type=float, help="sampling sigma for duration")
    parser.add_argument("--sigma_f0", default=1.0, type=float, help="sampling sigma for f0")
    parser.add_argument("--sigma_energy", default=1.0, type=float, help="sampling sigma for energy avg")
    parser.add_argument("--f0_mean", default=0.0, type=float)
    parser.add_argument("--f0_std", default=0.0, type=float)
    parser.add_argument("--energy_mean", default=0.0, type=float)
    parser.add_argument("--energy_std", default=0.0, type=float)
    parser.add_argument("--token_dur_scaling", default=1.00, type=float)
    parser.add_argument("--n_takes", default=1, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    infer(args.radtts_path, args.vocoder_path, args.vocoder_config_path,
          args.text_path, '', '',
          '', args.sigma, args.sigma_tkndur, args.sigma_f0,
          args.sigma_energy, args.f0_mean, args.f0_std, args.energy_mean,
          args.energy_std, args.token_dur_scaling, args.denoising_strength,
          args.n_takes, args.output_dir, args.use_amp, False, args.seed)
