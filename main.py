import hashlib
import requests
import argparse
from tqdm import tqdm
from os.path import isdir, isfile
from os import stat, makedirs
from colorama import Fore, Style, init as colorama_init
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
import torch
import re as regex
import time

API_ENDPOINT = "https://civitai.com/api/v1/models/"
image_size = 512
fp_half_precision = False


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


def get_sha256_checksum(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f, tqdm(
            desc='Comparing checksum',
            total=stat(filename).st_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
    ) as bar:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            bar.update(4096)
        return sha256_hash.hexdigest().upper()


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


def file_not_exist_or_can_override(filename: str, checksum: str):
    if not isfile(filename):
        return True
    print(f'File {Fore.RED}{filename}{Fore.RESET} already exist.')
    time.sleep(0.2)
    res = compare_checksum(filename, checksum)
    print(f'Checksums are {"identical" if res else "different"}.')
    if res:
        print('No need to download the checkpoint file. Continuing the process with existing one...')
        return False
    return get_user_decision()


def get_user_decision() -> bool:
    while True:
        user_decision = input(f'Type {Fore.RED}y/yes{Fore.RESET} to re-download and override the checkpoint file, '
                              f'{Fore.RED}n/no{Fore.RESET} to continue with existing checkpoint '
                              f'or {Fore.RED}a/abort{Fore.RESET} to exit the program.')
        user_decision = user_decision.lower().strip()
        if user_decision == 'y' or user_decision == 'yes':
            return True
        if user_decision == 'n' or user_decision == 'no':
            return False
        if user_decision == 'a' or user_decision == 'abort':
            exit(0)


def exit_with_error(err: str):
    print(Fore.RED, err, Fore.RESET)
    exit(0)


def compare_checksum(filename: str, checksum: str) -> bool:
    if not isfile(filename):
        return False
    return get_sha256_checksum(filename) == checksum


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

    if response['modelVersions'][0]['baseModel'].split()[-1] == '768':
        image_size = 768
    if response['modelVersions'][0]['files'][0]['metadata']['fp'] == 'fp16':
        fp_half_precision = True
    repo_name = regex.sub('[\\\/:*?"<>| ]*', '', response['name'])
    version_name = response['modelVersions'][0]['name'].replace(' ', '')
    checkpoint_name = response['modelVersions'][0]['files'][0]['name']

    dir_name = f'{model_id}_{repo_name}_{version_name}'
    checkpoint_name = f'{model_id}_{checkpoint_name}'
    checkpoint_format = response['modelVersions'][0]['files'][0]['metadata']['format']

    if request.status_code != 200:
        exit_with_error('The provided url is not valid.')
    if response['type'] == 'LORA':
        exit_with_error('Unsupported checkpoint type.')
    if checkpoint_format != 'SafeTensor':
        exit_with_error('Unsupported checkpoint format.')

    download_url = response['modelVersions'][0]['files'][0]['downloadUrl']
    checkpoint = f'checkpoints/{checkpoint_name}'
    remote_checksum = response['modelVersions'][0]['files'][0]['hashes']['SHA256']

    if file_not_exist_or_can_override(checkpoint, remote_checksum):
        while True:
            download(download_url, checkpoint)
            if not compare_checksum(filename=checkpoint, checksum=remote_checksum):
                print('Checksums are different. File most likely is corrupt.')
                if not get_user_decision():
                    break

    if not args.output:
        args.output = f'models/{dir_name}/'

    print('Note that the conversion process may take up to hours')
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=checkpoint,
        from_safetensors=True,
        image_size=image_size)

    if fp_half_precision:
        pipe.to(torch_device='cuda', torch_dtype=torch.float16)
    else:
        pipe.to(torch_device='cuda')

    pipe.save_pretrained(save_directory=args.output)
    print('Conversion is done!')
