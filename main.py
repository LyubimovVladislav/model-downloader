import requests
import argparse
from tqdm import tqdm
from os.path import isdir, isfile
from os import makedirs
from colorama import Fore, Style, init as colorama_init
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

API_ENDPOINT = "https://civitai.com/api/v1/models/"


def download(url: str, filename: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


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


def file_not_exist_or_can_override():
    if not isfile(checkpoint):
        return True
    print(f'{checkpoint} already exist. Override the file?')
    while True:
        user_decision = input('Type "y" to download and override the checkpoint '
                              'or "n" to continue with saved checkpoint')
        user_decision = user_decision.lower().strip()
        if user_decision == 'y':
            return True
        if user_decision == 'n':
            return False


def exit_with_error(err: str):
    print(Fore.RED, err, Style.RESET_ALL)
    exit(0)


if __name__ == "__main__":
    args = make_arg_parser()
    colorama_init()

    if not isdir('checkpoints'):
        makedirs('checkpoints')
    if not isdir('models'):
        makedirs('models')

    model_id = args.url[args.url.find('models/'):].split('/')[1]
    if model_id == '':
        exit_with_error('The provided url is not valid.')

    request = requests.get(API_ENDPOINT + model_id)
    response = request.json()

    if request.status_code != 200:
        exit_with_error('The provided url is not valid.')
    if response['type'] == 'LORA':
        exit_with_error('Unsupported checkpoint file.')

    download_url = response['modelVersions'][0]['files'][0]['downloadUrl']
    checkpoint = f'checkpoints/{model_id}.safetensors'

    if file_not_exist_or_can_override():
        download(download_url, checkpoint)

    if not args.output:
        args.output = f'models/{model_id}/'

    print('Note that the process may take up to hours')
    pipe = download_from_original_stable_diffusion_ckpt(checkpoint_path=checkpoint, from_safetensors=True)
    pipe.save_pretrained(save_directory=args.output)
