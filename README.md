<div align="center">
<h1>MetaDict</h1>

_Enabling dot notation and IDE autocompletion_

<p align="center">
<a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
<a href="#documentation">Documentation</a> •
  <a href="#competitors">Competitors</a> •
  <a href="#citation">Citation</a>
</p>

[![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI version](https://badge.fury.io/py/metadict.svg?dummy=unused)](https://badge.fury.io/py/metadict)
[![CircleCI](https://circleci.com/gh/LarsHill/metadict/tree/main.svg?style=shield)](https://circleci.com/gh/LarsHill/metadict/tree/main)
[![codecov](https://codecov.io/gh/LarsHill/metadict/branch/main/graph/badge.svg?token=XG4UDWF8RE)](https://codecov.io/gh/LarsHill/metadict)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

**MetaDict** is designed to behave exactly like a `dict` while enabling (nested) attribute-style key access/assignment with IDE autocompletion support. 

Many libraries claim to do the same, but fail in different ways (see <a href="#competitors">Competitors</a>). 

## Installation

```bash
$ pip install metadict
```
## Features

- Attribute-style key access and assignment (dot notation) with IDE autocompletion support
   ```python
   from metadict import MetaDict
   
   cfg = MetaDict()
   cfg.optimizer = 'Adam'
   print(cfg.optimizer)
   >> Adam
   ```
   ![autocompletion demo](/autocompletion.png?raw=true "Autocompletion demo")
- Nested key assignment similar to `defaultdict` from `collections`
   ```python
   cfg = MetaDict(nested_assignment=True)
   cfg.model.type = 'Transformer' 
   print(cfg.model.type)
   >> Transformer
   
   # or restrict nested assignment via context manager
   cfg = MetaDict()
   with cfg.enabling_nested_assignment() as cfg:
       cfg.model.type = 'Transformer'
   cfg.new_model.type = 'MLP'
   >> AttributeError: 'MetaDict' object has no attribute 'new_model'
   ```
- Is a `dict`
   ```python
   dict_config = {'model': 'Transformer',
                  'optimizer': 'Adam'}    
   cfg = MetaDict(dict_config)
   print(isinstance(cfg, dict))
   >> True
   print(cfg == dict_config)
   >> True
   ```
- Inbuilt `json` support
   ```python
   import json
       
   cfg = MetaDict({'model': 'Transformer'})
   print(json.loads(json.dumps(cfg)))
   >> {'model': 'Transformer'}
   ```
- Recursive conversion to `dict`
   ```python  
   cfg = MetaDict({'models': [{'name': 'Transformer'}, {'name': 'MLP'}]})
   print(cfg.models[0].name)
   >> Transformer
   
   cfg_dict = cfg.to_dict()
   print(type(cfg_dict['models'][0]['name']))
   >> dict
   ```
- No namespace conflicts with inbuilt methods like `items()`, `update()`, etc.
   ```python  
   cfg = MetaDict()
   # Key 'items' is assigned as in a normal dict, but a UserWarning is raised
   cfg.items = [1, 2, 3]
   >> UserWarning: 'MetaDict' object uses 'items' internally. 'items' can only be accessed via `obj['items']`.
   print(cfg)
   >> {'items': [1, 2, 3]}
   print(cfg['items'])
   >> [1, 2, 3]
   
   # But the items method is not overwritten!
   print(cfg.items)
   >> <bound method Mapping.items of {'items': [1, 2, 3]}>
   print(list(cfg.items()))
   >> [('items', [1, 2, 3])]
   ```
- References are preserved
   ```python
   params = [1, 2, 3]    
   cfg = MetaDict({'params': params})
   print(cfg.params is params)
   >> True
   
   model_dict = {'params': params}
   cfg = MetaDict(model=model_dict)
   print(cfg.model.params is params)
   >> True
   
   # Note: dicts are recursively converted to MetaDicts, thus...
   print(cfg.model is model_dict)
   >> False
   print(cfg.model == model_dict)
   >> True~~~~
   ```

## Documentation

Check the [Test Cases](https://github.com/LarsHill/metadict/blob/main/tests/test_metadict.py) for a complete overview of all **MetaDict** features.


## Competitors
- [Addict](https://github.com/mewwts/addict)
  - No key autocompletion in IDE
  - Nested key assignment cannot be turned off
  - Newly assigned `dict` objects are not converted to support attribute-style key access
  - Shadows inbuilt type `Dict`
- [Prodict](https://github.com/ramazanpolat/prodict)
  - No key autocompletion in IDE without defining a static schema (similar to `dataclass`)
  - No recursive conversion of `dict` objects when embedded in `list` or other inbuilt iterables
- [AttrDict](https://github.com/bcj/AttrDict)
  - No key autocompletion in IDE
  - Converts `list` objects to `tuple` behind the scenes
- [Munch](https://github.com/Infinidat/munch)
  - Inbuilt methods like `items()`, `update()`, etc. can be overwritten with `obj.items = [1, 2, 3]` 
  - No recursive conversion of `dict` objects when embedded in `list` or other inbuilt iterables
- [EasyDict](https://github.com/makinacorpus/easydict)
  - Only strings are valid keys, but `dict` accepts all hashable objects as keys
  - Inbuilt methods like `items()`, `update()`, etc. can be overwritten with `obj.items = [1, 2, 3]`
  - Inbuilt methods don't behave as expected: `obj.pop('unknown_key', None)` raises an `AttributeError`


## Citation

```
@article{metadict,
  title = {MetaDict - Enabling dot notation and IDE autocompletion},
  author = {Hillebrand, Lars},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LarsHill/metadict}},
}
```
