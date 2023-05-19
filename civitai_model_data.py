from dataclasses import dataclass

import re as regex
from typing import Optional

import requests

from exit_with_error import exit_with_error

API_GET_BY_ID_ENDPOINT = 'https://civitai.com/api/v1/models/'
API_SEARCH_BY_NAME_ENDPOINT = 'https://civitai.com/api/v1/models'


# noinspection SpellCheckingInspection
@dataclass
class CivitaiModelData:

    def __init__(self, url, response=None):
        self.model_id = url.split('/')[0]
        if self.model_id == '':
            exit_with_error('The provided url is not valid.')
        if not response:
            request = requests.get(API_GET_BY_ID_ENDPOINT + self.model_id)
            if request.status_code != 200:
                exit_with_error('The provided url is not valid.')
            response = request.json()
        self.image_size = 768 if response['modelVersions'][0]['baseModel'].split()[-1] == '768' else 512
        self.fp_half_precision = response['modelVersions'][0]['files'][0]['metadata']['fp'] == 'fp16'
        self.repo_name = regex.sub('[\\\/:*?"<>| ]*', '', response['name'])
        self.version_name = response['modelVersions'][0]['name'].replace(' ', '')
        self.checkpoint_name = response['modelVersions'][0]['files'][0]['name']

        self.checkpoint_name = f'{self.model_id}_{self.checkpoint_name}'
        self.checkpoint_format = response['modelVersions'][0]['files'][0]['metadata']['format']

        self.download_url = response['modelVersions'][0]['files'][0]['downloadUrl']
        self.checkpoint = f'checkpoints/{self.checkpoint_name}'
        self.remote_checksum = response['modelVersions'][0]['files'][0]['hashes']['SHA256']
        self.remote_size_bytes = int(response['modelVersions'][0]['files'][0]['sizeKB'] * 1024)
        self.type = response['type']
        self.base: Optional[CivitaiModelData] = None
        try:
            self.base_model_name = response['modelVersions'][0]['images'][0]['meta']['Model']
        except Exception as e:
            pass

    def load_lora_base_model_info(self, url=None):
        if not self.base_model_name:
            return None
        request = requests.get(
            API_SEARCH_BY_NAME_ENDPOINT,
            params={"limit": 1, "query": url if url else self.base_model_name})
        if request.status_code != 200:
            return None
        response = request.json()
        try:
            self.base = CivitaiModelData(url=response['items'][0]['modelVersions'][0]['files'][0]['downloadUrl'],
                                         response=response['items'][0])
        except Exception as e:
            return None
        return self.base
