import hashlib
import time
from http.client import IncompleteRead

import requests
from colorama import Fore
from tqdm import tqdm
from os import stat
from os.path import isfile

from user_decision import get_user_decision


def is_file_exist(file: str) -> bool:
    if isfile(file):
        print(f'File {Fore.RED}{file}{Fore.RESET} already exist.')
        return True
    return False


def is_allowed_override(file: str, checksum: str) -> bool:
    time.sleep(0.2)
    res = is_checksum_equal(file, checksum)
    print(f'Checksums are {"identical" if res else "different"}.')
    if res:
        print('No need to download the checkpoint file. Continuing the process with existing one...')
        return False
    return get_user_decision()


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
            bar.update(len(byte_block))
        return sha256_hash.hexdigest()


def is_checksum_equal(filename: str, checksum: str) -> bool:
    if not isfile(filename):
        return False
    return get_sha256_checksum(filename).upper() == checksum.upper()


def compare_sizes(file: str, remote_size: int) -> int:
    if stat(file).st_size > remote_size:
        return 1
    elif stat(file).st_size < remote_size:
        return -1
    else:
        return 0


def download_from_pos(url: str, filename: str, resume_pos: int = 0):
    resp = requests.get(
        url,
        stream=True,
        headers={'Range': f'bytes={resume_pos}-'} if resume_pos else None
    )
    total = int(resp.headers.get('content-length', 0))
    mode = 'ab' if resume_pos else 'wb'
    with open(filename, mode) as file, tqdm(
            initial=resume_pos,
            desc=filename,
            total=total + resume_pos,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def start_download(download_url: str, file: str):
    download_from_pos(download_url, file)


def resume_download(download_url: str, file: str):
    download_from_pos(download_url, file, stat(file).st_size)


def download(download_url: str, file: str, remote_size: int, remote_checksum: str):
    if is_file_exist(file) and not is_allowed_override(file, remote_checksum):
        return
    while True:
        try:
            start_download(download_url, file)
        except IncompleteRead as e:
            pass
        while compare_sizes(file, remote_size) < 0:
            try:
                resume_download(download_url, file)
            except IncompleteRead as e:
                pass
        if is_checksum_equal(file, remote_checksum):
            break
        print('File is corrupted. Restarting download...')
    print('The download is complete.')
