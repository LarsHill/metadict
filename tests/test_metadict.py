from typing import Dict, List, Tuple, Any
import copy
import json
import pickle

import pytest

from metadict import MetaDict
from metadict.metadict import complies_variable_syntax  # , _warning


@pytest.fixture
def config() -> Dict:
    dropout = 0.1
    special_tokens = [['<NUM>', '<YEAR>']]
    tokenizer_params = {'special_tokens': special_tokens}
    return {'epochs': 10,
            'model': [{'type_': 'mlp',
                       'hidden_dims': [128, 256],
                       'dropout': dropout},
                      {'type_': 'lstm',
                       'hidden_dim': 128,
                       'dropout': dropout}],
            'optimizer': {'type_': 'adam',
                          'lr': 1.e-3,
                          'weight_decay': True},
            'tokenizer_1': tokenizer_params,
            'tokenizer_2': tokenizer_params}


@pytest.fixture
def list_of_tuples() -> List[Tuple]:
    return [('a', [1, 2, 3]), ('b', {'c': 6})]


def test_init_from_dict(config: Dict):
    cfg = MetaDict(config)
    assert cfg == config


def test_init_from_kwargs(config: Dict):
    cfg = MetaDict(**config)
    assert cfg == config


def test_init_from_list_of_tuples(list_of_tuples: List[Tuple]):
    cfg = MetaDict(list_of_tuples)
    assert cfg == dict(list_of_tuples)


def test_from_keys(config: Dict):
    cfg = MetaDict.fromkeys(config.keys())
    config = dict.fromkeys(config.keys())
    assert cfg == config


def test_get_item(config: Dict):
    cfg = MetaDict(config)
    assert cfg['model'][0]['type_'] == config['model'][0]['type_']
    with pytest.raises(KeyError) as _:
        _ = cfg['some_missing_key']


def test_get_attr(config: Dict):
    cfg = MetaDict(config)
    assert cfg.model[0].type_ == config['model'][0]['type_']
    with pytest.raises(AttributeError) as _:
        _ = cfg.some_missing_attr


def test_set_item(config: Dict):
    cfg = MetaDict(config)
    cfg[('new', 'key')] = 'new_value'
    assert cfg[('new', 'key')] == 'new_value'


def test_set_attr(config: Dict):
    cfg = MetaDict(config)
    cfg.new_attr = {'new_key': 'new_value'}
    assert cfg.new_attr.new_key == 'new_value'


def test_del_item(config: Dict):
    cfg = MetaDict(config)
    del cfg['model'][0]['type_']
    del config['model'][0]['type_']
    assert cfg == config
    with pytest.raises(KeyError) as _:
        del cfg['some_missing_key']


def test_del_attr(config: Dict):
    cfg = MetaDict(config)
    del cfg.model[0].type_
    del config['model'][0]['type_']
    assert cfg == config
    with pytest.raises(AttributeError) as _:
        del cfg.some_missing_attr


def test_get(config: Dict):
    cfg = MetaDict(config)
    assert cfg.get('model') == config.get('model')
    assert cfg.get('some_new_key', 100) == 100


def test_pop(config: Dict):
    cfg = MetaDict(config)
    assert cfg.pop('model') == config.pop('model')
    assert cfg.pop('some_new_key', 100) == 100


def test_get_nested(config: Dict):
    cfg = MetaDict(config)
    cfg.enable_nested_access()
    assert cfg.get('tokenizer_1.special_tokens') == config['tokenizer_1'].get('special_tokens')


def test_pop_nested(config: Dict):
    cfg = MetaDict(config)
    cfg.enable_nested_access()
    assert cfg.pop('tokenizer_1.special_tokens') == config['tokenizer_1'].pop('special_tokens')
    assert cfg == config


def test_keys(config: Dict):
    cfg = MetaDict(config)
    assert cfg.keys() == config.keys()


def test_values(config: Dict):
    cfg = MetaDict(config)
    assert list(cfg.values()) == list(config.values())


def test_items(config: Dict):
    cfg = MetaDict(config)
    assert cfg.items() == config.items()


def test_update(config: Dict):
    cfg = MetaDict(config)
    cfg.update({'a': 1, 'b': 2})
    config.update({'a': 1, 'b': 2})
    assert cfg == config


def test_copy(config: Dict):
    cfg = MetaDict(config)
    assert copy.copy(cfg) is not cfg
    assert cfg.copy() == copy.copy(cfg)


def test_str(config: Dict):
    cfg = MetaDict(config)
    assert str(cfg) == str(config)


def test_repr(config: Dict):
    cfg = MetaDict(config)
    assert repr(cfg) == repr(config)


def test_dir(config: Dict):
    cfg = MetaDict(config)
    assert set(dir(cfg)) == set(dir(MetaDict) + [key for key in cfg._data.keys() if complies_variable_syntax(key)])


def test_nested_assignment_default(config: Dict):
    cfg = MetaDict(config)
    with pytest.raises(AttributeError) as _:
        cfg.x.y.z = 100
    assert cfg.nested_assignment is False


def test_nested_assignment(config: Dict):
    cfg = MetaDict(config)
    cfg.enable_nested_assignment()
    assert cfg.nested_assignment is True
    cfg.x.y.z = 100
    assert cfg.x.y.z == 100
    cfg.disable_nested_assignment()
    assert cfg.nested_assignment is False


def test_nested_access_default(config: Dict):
    cfg = MetaDict(config)
    cfg.enable_nested_assignment()
    cfg.x.y.z = 100
    cfg.disable_nested_assignment()
    assert cfg.nested_access is False
    with pytest.raises(KeyError) as _:
        _ = cfg['x.y.z']


def test_nested_access(config: Dict):
    cfg = MetaDict(config)
    cfg.enable_nested_assignment()
    cfg.x.y.z = 100
    cfg.enable_nested_access()
    assert cfg.get('x.y.z') == 100
    assert cfg.nested_access is True
    cfg.disable_nested_access()
    assert cfg.nested_access is False


def test_json(config: Dict):
    cfg = MetaDict(config)
    cfg_json = json.loads(json.dumps(cfg))
    assert cfg_json == config
    assert type(cfg_json) == dict


def test_pickle(config: Dict):
    cfg = MetaDict(config)
    cfg_pickle = pickle.loads(pickle.dumps(cfg))
    assert cfg_pickle == cfg
    assert isinstance(cfg_pickle, MetaDict)


def test_is_instance_dict(config: Dict):
    cfg = MetaDict(config)
    assert isinstance(cfg, dict)


def test_to_dict(config: Dict):
    cfg = MetaDict(config)
    cfg_dict = cfg.to_dict()
    assert cfg_dict == config
    assert type(cfg_dict) == dict
    assert type(cfg_dict['model'][0]) == dict


def test_to_object(config: Dict):
    cfg = MetaDict(config)
    cfg_dict = MetaDict.to_object(cfg)
    assert cfg_dict == config
    assert type(cfg_dict) == dict
    assert type(cfg_dict['model'][0]) == dict


def test_references(config: Dict):
    cfg = MetaDict(config)
    assert cfg.tokenizer_1 is cfg.tokenizer_2
    assert cfg.tokenizer_1.special_tokens is config['tokenizer_1']['special_tokens']


def test_append_dict_to_list(config: Dict):
    cfg = MetaDict(config)
    cfg.model.append({'type_': 'gru'})
    assert isinstance(cfg.model[-1], MetaDict)


def test_init_type_checks():
    with pytest.raises(TypeError) as _:
        MetaDict(config, nested_access='wrong_type')

    with pytest.raises(TypeError) as _:
        MetaDict(config, nested_assignment=999)


def test_warning_protected_key():
    with pytest.warns(UserWarning) as warn_inf:
        MetaDict(items=[1, 2, 3])
        assert str(warn_inf.list[0].message) == "'MetaDict' object uses 'items' internally. " \
                                                "'items' can only be accessed via `obj['items']`."


# def test_warning(capsys):
#     _warning('Some Warning.')
#     captured = capsys.readouterr()
#     assert captured.out == 'UserWarning: Some Warning.\n'


def test_no_dot_in_key_if_nested_access():
    cfg = MetaDict({'some.dotted.key': 100})
    with pytest.raises(ValueError):
        cfg.enable_nested_access()

    del cfg['some.dotted.key']
    cfg.enable_nested_access()
    with pytest.raises(ValueError):
        cfg['some.dotted.key'] = 100


@pytest.mark.parametrize("name, expected", [('models', True), ('text_100', True), ('1name', False), ('&%?=99', False),
                                            ('100', False), (100, False), ((1, 2, 3), False)])
def test_complies_variable_syntax(name: Any, expected: bool):
    assert complies_variable_syntax(name) == expected
