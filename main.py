import requests
import argparse
from os.path import isdir
from os import makedirs
from colorama import Fore, init as colorama_init
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
import torch
from diffusers import StableDiffusionPipeline

from civitai_model_data import CivitaiModelData
from convert_lora_safetensor_to_diffusers import convert
from downloader import download

API_ENDPOINT = 'https://civitai.com/api/v1/models/'


def make_arg_parser():
    parser = argparse.ArgumentParser(
        prog='CivitAI model downloader',
        description='Downloads a checkpoint from CivitAI and outputs a model to use in diffusers.',
        epilog='Text at the bottom of help')
    parser.add_argument(
        '--url', '-u',
        type=str,
        required=True,
        default=None,
        help='Url to the CivitAI checkpoint to download and convert.')
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to the output model.')
    return parser.parse_args()


def exit_with_error(err: str):
    print(Fore.RED, err, Fore.RESET)
    exit(0)


def create_folders():
    if not isdir('checkpoints'):
        makedirs('checkpoints')
    if not isdir('models'):
        makedirs('models')


def load_lora(data, output):
    file = 'checkpoints/stable_diffusion_1_5.safetensors'
    download('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors',
             file=file,
             remote_checksum='1a189f0be69d6106a48548e7626207dddd7042a418dbf372cefd05e0cdba61b6')
    res_dir = 'models/stable_diffusion_1_5'
    base = StableDiffusionPipeline.from_ckpt(file)
    base.save_pretrained(save_directory=res_dir)
    pipe = convert(res_dir, data.checkpoint, 'lora_unet', 'lora_te', '0.75',
                   torch.float16 if data.fp_half_precision else torch.float32)
    pipe = pipe.to(torch_dtype=torch.float16 if data.fp_half_precision else torch.float32)
    pipe.save_pretrained(output, safe_serialization=True)


def civitai_link(model_id: str, output: str = None):
    request = requests.get(API_ENDPOINT + model_id)
    if request.status_code != 200:
        exit_with_error('The provided url is not valid.')
    response = request.json()

    data = CivitaiModelData(response, model_id)
    dir_name = f'{model_id}_{data.repo_name}_{data.version_name}'

    download(download_url=data.download_url, file=data.checkpoint, remote_checksum=data.remote_checksum)

    if not output:
        output = f'models/{dir_name}/'

    if data.type == 'LORA':
        load_lora(data=data, output=output)
        return

    print('Note that the conversion process may take up to hours')
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=data.checkpoint,
        from_safetensors=data.checkpoint_format == 'SafeTensor',
        image_size=data.image_size,
        device='cuda'
    )

    if data.fp_half_precision:
        pipe.to(torch_dtype=torch.float16)

    pipe.save_pretrained(save_directory=output)
    print('Conversion is done!')


def hf_link(url: str, output: str = None):
    pipe = StableDiffusionPipeline.from_ckpt(url)
    dir_name = url.split('/')[-1].replace('.safetensors', '').replace('.ckpt', '')
    if output is None:
        output = f'models/{dir_name}'
    pipe.save_pretrained(save_directory=output)
    print('Conversion is done!')


def main():
    args = make_arg_parser()
    colorama_init()
    create_folders()
    remote_repo_type = None

    for prefix in ['https://huggingface.co/', 'huggingface.co/', 'hf.co/', 'https://hf.co/']:
        if args.url.startswith(prefix):
            remote_repo_type = 'hf'
            break
    for prefix in ['https://civitai.com/models/', 'civitai.com/models/']:
        if args.url.startswith(prefix):
            remote_repo_type = 'civit'
            args.url = args.url[len(prefix):]
            break

    if remote_repo_type is None:
        exit_with_error('The provided url is not valid.')
    elif remote_repo_type == 'hf':
        hf_link(args.url, args.output)
    elif remote_repo_type == 'civit':
        model_id = args.url.split('/')[0]
        if model_id == '':
            exit_with_error('The provided url is not valid.')
        civitai_link(model_id, args.output)


if __name__ == "__main__":
    main()
