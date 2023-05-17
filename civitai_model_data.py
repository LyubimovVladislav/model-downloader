from dataclasses import dataclass

import re as regex


# noinspection SpellCheckingInspection
@dataclass
class CivitaiModelData:

    def __init__(self, response, model_id):
        self.image_size = 768 if response['modelVersions'][0]['baseModel'].split()[-1] == '768' else 512
        self.fp_half_precision = response['modelVersions'][0]['files'][0]['metadata']['fp'] == 'fp16'
        self.repo_name = regex.sub('[\\\/:*?"<>| ]*', '', response['name'])
        self.version_name = response['modelVersions'][0]['name'].replace(' ', '')
        self.checkpoint_name = response['modelVersions'][0]['files'][0]['name']

        self.checkpoint_name = f'{model_id}_{self.checkpoint_name}'
        self.checkpoint_format = response['modelVersions'][0]['files'][0]['metadata']['format']

        self.download_url = response['modelVersions'][0]['files'][0]['downloadUrl']
        self.checkpoint = f'checkpoints/{self.checkpoint_name}'
        self.remote_checksum = response['modelVersions'][0]['files'][0]['hashes']['SHA256']
        self.remote_size_bytes = int(response['modelVersions'][0]['files'][0]['sizeKB'] * 1024)
        self.type = response['type']
