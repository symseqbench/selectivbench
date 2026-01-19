# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import yaml

class Config:

    def __init__(self, config):
        self.config = config

    @staticmethod
    def parse_config_from_args(args):
        with open(args['config'], 'r') as f:    
            config = yaml.safe_load(f)

        assert config is not None, 'Provided config seems to be empty' 

        for k, v in args.items():
            if v is not None:
                if k in config.keys():
                    print(f'CLI config overwrite for "{k}"!')
                config[k] = v

        return Config(config)

    @staticmethod
    def parse_config_from_path(path):
        with open(path, 'r') as f:    
            config = yaml.safe_load(f)
        
        assert config is not None, 'Provided config seems to be empty' 

        return Config(config)

    def asdict(self):
        as_dict = {}
        for k, v in self.items():
            if isinstance(v, Config):
                as_dict[k] = v.as_dict()
            else:
                as_dict[k] = v
        return as_dict

    def print_config(self):
        max_key_length = max([len(k) for k in self.config.keys()])
        for key in self.config:
            print(key.ljust(max_key_length, '-'), str(self.config[key]).ljust(100, '-'))

    def assert_has_key(self, key):
        if not self.has_key(key):
            error_msg = f'Config is missing key "{key}".'
            raise ValueError(error_msg)
     
    def has_key(self, key):
        return key in self.config.keys()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            default = key[1]
            key = key[0]
            return self.__get_item_with_default(key, default)
        else:
            return self.__get_item(key)
    
    def __get_item(self, key):
        self.assert_has_key(key)
        item = self.__parse_item(self.config[key])
        return item

    def __get_item_with_default(self, key, default):
        if self.has_key(key):    
            item = self.__parse_item(self.config[key])
            return item
        else:
            return self.__parse_item(default)

    def __parse_item(self, item):
        if isinstance(item, dict):
            return Config(item)
        else:
            return item

    def __setitem__(self, key, value):
        self.config[key] = value
    
    def update(self, dict, prefix=''):
        for k, v in dict.items():
            self.config[f'{prefix}{k}'] = v

    def items(self):
        return self.config.items()

    def keys(self):
        return self.config.keys()