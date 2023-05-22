from dataclasses import dataclass

import re as regex
from os.path import dirname
from typing import Optional

import requests

from exit_with_error import exit_with_error

API_GET_BY_ID_ENDPOINT = 'https://civitai.com/api/v1/models/'
API_SEARCH_BY_NAME_ENDPOINT = 'https://civitai.com/api/v1/models'


# noinspection SpellCheckingInspection
@dataclass
class CivitaiModelData:

    def __init__(self, model_id, response=None):
        regex_pattern = regex.escape('\/:*?"<>| .')
        regex_pattern = f'[{regex_pattern}]*'
        self.model_id = model_id
        if model_id == '':
            raise ValueError('The provided url is not valid.')
        if not response:
            request = requests.get(API_GET_BY_ID_ENDPOINT + model_id)
            if request.status_code != 200:
                raise ValueError('The provided url is not valid.')
            response = request.json()
        self.image_size = 768 if response['modelVersions'][0]['baseModel'].split()[-1] == '768' else 512
        self.fp_half_precision = response['modelVersions'][0]['files'][0]['metadata']['fp'] == 'fp16'
        self.repo_name = regex.sub(regex_pattern, '', response['name'])
        self.version_name = regex.sub(regex_pattern, '', response['modelVersions'][0]['name'])
        self.checkpoint_name = response['modelVersions'][0]['files'][0]['name']

        self.checkpoint_name = f'{self.model_id}_{self.checkpoint_name}'
        self.checkpoint_format = response['modelVersions'][0]['files'][0]['metadata']['format']

        self.download_url = response['modelVersions'][0]['files'][0]['downloadUrl']
        self.checkpoint = f'{dirname(__file__)}/checkpoints/{self.checkpoint_name}'
        self.remote_checksum = response['modelVersions'][0]['files'][0]['hashes']['SHA256']
        self.remote_size_bytes = int(response['modelVersions'][0]['files'][0]['sizeKB'] * 1024)
        self.type = response['type']
        self.base: Optional[CivitaiModelData] = None
        try:
            self.base_model_name = response['modelVersions'][0]['images'][0]['meta']['Model']
        except Exception as e:
            pass

    def load_lora_base_model_info(self):
        if not self.base_model_name:
            return None
        request = requests.get(
            API_SEARCH_BY_NAME_ENDPOINT,
            params={"limit": 1, "query": self.base_model_name})
        if request.status_code != 200:
            return None
        response = request.json()
        try:
            self.base = CivitaiModelData(model_id=response['items'][0]['id'],
                                         response=response['items'][0])
        except Exception as e:
            return None
        return self.base

    def load_lora_from_url(self, url):
        if url is None:
            raise ValueError('Url link cant be empty!')
        for prefix in ['https://civitai.com/models/', 'civitai.com/models/']:
            if url.startswith(prefix):
                model_id = url[len(prefix):].split('/')[0]
                self.base = CivitaiModelData(model_id=model_id)
                return self.base
        return None
