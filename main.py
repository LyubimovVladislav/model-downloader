import argparse
import torch
from os.path import isdir, dirname
from os import makedirs
from colorama import init as colorama_init
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from diffusers import StableDiffusionPipeline

from civitai_model_data import CivitaiModelData
from convert_lora_safetensor_to_diffusers import convert
from downloader import download
from exit_with_error import exit_with_error
from user_decision import ask_for_base_model_link


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
    parser.add_argument(
        '--alpha', '-a',
        metavar="[0-1]",
        type=int,
        default=0.75,
        help='The merging strength of LoRA weights. Can exceed 1 if you want to merge amplified LoRA weights')
    return parser.parse_args()


def create_folders():
    if not isdir(f'{dirname(__file__)}/checkpoints'):
        makedirs(f'{dirname(__file__)}/checkpoints')
    if not isdir(f'{dirname(__file__)}/models'):
        makedirs(f'{dirname(__file__)}/models')


def load_lora(lora_data, alpha, output):
    base_model_data = lora_data.load_lora_base_model_info()
    while True:
        if base_model_data:
            break
        base_model_data = lora_data.load_lora_base_model_info(ask_for_base_model_link())

    download(base_model_data.download_url,
             file=base_model_data.checkpoint,
             remote_checksum=base_model_data.remote_checksum)
    # base.save_pretrained(save_directory=res_dir)
    pipe = convert(base_model_data, lora_data, alpha=alpha)
    pipe = pipe.to(torch_dtype=torch.float16 if base_model_data.fp_half_precision else torch.float64, device='cuda')
    pipe.save_pretrained(output, safe_serialization=True)


def civitai_link(model_id: str, alpha, output: str = None):
    data = CivitaiModelData(model_id)
    dir_name = f'{data.model_id}_{data.repo_name}_{data.version_name}'

    download(download_url=data.download_url, file=data.checkpoint, remote_checksum=data.remote_checksum)

    if not output:
        output = f'{dirname(__file__)}/models/{dir_name}/'

    if data.type == 'LORA':
        data.load_lora_base_model_info()
        load_lora(lora_data=data, alpha=alpha, output=output)
        return

    print('Note that the conversion process may take up to hours')
    pipe = StableDiffusionPipeline.from_ckpt(
        pretrained_model_link_or_path=data.checkpoint,
        use_safetensors=data.checkpoint_format == 'SafeTensor',
        image_size=data.image_size,
        extract_ema=True,
        torch_dtype=torch.float16 if data.fp_half_precision else torch.float64
    )

    pipe.to(torch_device='cuda')

    pipe.save_pretrained(save_directory=output)
    print('Conversion is done!')


def hf_link(url: str, output: str = None):
    pipe = StableDiffusionPipeline.from_ckpt(url, extract_ema=True)
    dir_name = url.split('/')[-1].replace('.safetensors', '').replace('.ckpt', '')
    if output is None:
        output = f'{dirname(__file__)}/models/{dir_name}'
    pipe.save_pretrained(save_directory=output)
    print('Conversion is done!')


def main():
    args = make_arg_parser()
    if args.alpha < 0:
        exit_with_error('Alpha cant be negative')
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
            args.url = args.url[len(prefix):].split('/')[0]
            break

    if remote_repo_type is None:
        exit_with_error('The provided url is not valid.')
    elif remote_repo_type == 'hf':
        hf_link(args.url, args.output)
    elif remote_repo_type == 'civit':
        civitai_link(args.url, args.alpha, args.output)


if __name__ == "__main__":
    main()
