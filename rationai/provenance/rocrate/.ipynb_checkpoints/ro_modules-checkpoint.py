from pathlib import Path
import os

from rocrate.model.contextentity import ContextEntity
from rocrate.model.creativework import CreativeWork
from rocrate.model.data_entity import DataEntity
from rocrate.model.file import File


class WSI_Collection(DataEntity):
    def __init__(self, crate, source=None, properties=None):
        if properties is None:
            properties = {}
        super().__init__(crate, os.path.basename(source), properties)
        crate.root_dataset['mentions'] += [self]
        
    def _empty(self):
        return {
            '@id': self.id,
            '@type': 'Collection',
            'hasPart': [],
            'mainEntity': {}
        }


class HistopatScript(File):    
    TYPES = ['File', 'SoftwareSourceCode']
    
    def __init__(self, crate, commit_hash, local_rep_path, properties=None, **kwargs):
        GIT_URL = 'https://github.com/RationAI/crc_ml-provenance/blob'
        
        script_path = Path(local_rep_path)
        script_path = str(Path(*script_path.parts[script_path.parts.index('rationai'):]))
        url = f'{GIT_URL}/{commit_hash}/{script_path}'
        
        if properties is None:
            properties = {}
        if 'url' not in properties:
            properties['url'] = url
            
        super().__init__(crate=crate, source=url, properties=properties, **kwargs)
    
    def _empty(self):
        return {
            '@id': self.id,
            '@type': self.TYPES[:],
            'name': None,
            'url': None
        }
    
    @property
    def name(self):
        return self.get('name')
    
    @name.setter
    def name(self, name):
        self['name'] = name
    
    @property
    def url(self):
        return self.get('url')
        
    @url.setter
    def url(self, url):
        self['url'] = url


class HistopatEntity(ContextEntity, CreativeWork):
    def _empty(self):
        return {
            '@id': self.id,
            '@type': 'CreateAction',
            'instrument': None,
            'name': None,
            'object': [],
            'result': []
        }
    
    @property
    def name(self):
        return self.get('name')
    
    @name.setter
    def name(self, name):
        self['name'] = name
    
    @property
    def instrument(self):
        return self.get('instrument')
        
    @instrument.setter
    def instrument(self, instrument):
        self['instrument'] = instrument