import contextlib
import keyword
import re
import warnings
from typing import MutableMapping, Dict, Iterator, TypeVar, Iterable, Optional, Mapping, Any, Tuple


def _warning(message, category=UserWarning, filename='', lineno=-1, file=None, line=''):
    """Monkey patch `warnings` to show UserWarning without the line information of warnings call."""
    msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
    print(f'{msg.category.__name__}: {msg.message}')


warnings.showwarning = _warning

KT = TypeVar('KT')
VT = TypeVar('VT')

# regex to enforce python variable/attribute syntax
ALLOWED_VAR_SYNTAX = re.compile(r'[a-zA-Z_]\w*')


def complies_variable_syntax(name: Any) -> bool:
    """Checks whether a given object is a string which complies the python variable syntax."""
    if not isinstance(name, str) or keyword.iskeyword(name):
        return False
    name_cleaned = ''.join(re.findall(ALLOWED_VAR_SYNTAX, name))
    return name_cleaned == name


class MetaDict(MutableMapping[KT, VT], dict):
    """Class that extends `dict` to access and assign keys via attribute dot notation.

    Example:
        d = MetaDict({'foo': {'bar': [{'a': 1}, {'a': 2}]}})
        print(d.foo.bar[1].a)
        >> 2
        print(d["foo"]["bar"][1]["a"])
        >> 2

    `MetaDict` inherits from MutableMapping to avoid overwriting all `dict` methods.
    In addition, it inherits from `dict` to pass the quite common `isinstance(obj, dict) check.
    Also, inheriting from `dict` enables json encoding/decoding without a custom encoder.
    """

    def __init__(
            self,
            *args,
            nested_assignment: bool = False,
            **kwargs
    ) -> None:

        # check that 'nested_assignment' is  of type bool
        if not isinstance(nested_assignment, bool):
            raise TypeError(f"Keyword argument 'nested_assignment' must be an instance of type 'bool'")

        # init internal attributes and data store
        self.__dict__['_data']: Dict[KT, VT] = {}
        self.__dict__['_nested_assignment'] = nested_assignment
        self.__dict__['_parent'] = kwargs.pop('_parent', None)
        self.__dict__['_key'] = kwargs.pop('_key', None)
        self.__dict__['_memory_map']: Dict[KT, VT] = {}

        # update state of data store
        self.update(*args, **kwargs)

        # call `dict` constructor with stored data to enable object encoding (e.g. `json.dumps()`) that relies on `dict`
        dict.__init__(self, self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[KT]:
        return iter(self._data)

    def __setitem__(self, key: KT, value: VT) -> None:
        # show a warning if the assigned key or attribute is used internally (e.g `items`, `keys`, etc.)
        try:
            self.__getattribute__(key)
            key_is_protected = True
        except (AttributeError, TypeError):
            key_is_protected = False
        if key_is_protected:
            warnings.warn(f"'{self.__class__.__name__}' object uses '{key}' internally. "
                          f"'{key}' can only be accessed via `obj['{key}']`.")

        # set key recursively
        self._data[key] = self.from_object(value)

        # update parent when nested keys or attributes are assigned
        parent = self.__dict__.pop('_parent', None)
        key = self.__dict__.get('_key', None)
        if parent is not None:
            parent[key] = self._data

    def __getitem__(self, key: KT) -> VT:
        try:
            value = self._data[key]
        except KeyError:
            if self.nested_assignment:
                return self.__missing__(key)
            raise

        # if retrieved value is of builtin sequence type we check whether it contains an object of type Mapping
        # (except MetaDict type objects).
        # if True we call __setitem__ on the retrieved key-value pair to make sure
        # all nested dicts are recursively converted to MetaDict objects.
        # this is necessary if e.g. a list type attribute/key is appended by a normal dict object.
        # oo dynamically convert the appended dict, we call __setitem__ again before the actual object retrieval.
        if isinstance(value, (list, set, tuple)) and MetaDict._contains_mapping(value, ignore=self.__class__):
            self[key] = value
            value = self._data[key]

        return value

    def __missing__(self, key: KT) -> 'MetaDict':
        return self.__class__(_parent=self, _key=key, nested_assignment=self._nested_assignment)

    def __delitem__(self, key: KT) -> None:
        del self._data[key]

    def __setattr__(self, attr: str, val: VT) -> None:
        self[attr] = val

    def __getattr__(self, key: KT) -> VT:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'") from None

    def __delattr__(self, key: KT) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'") from None

    def __str__(self) -> str:
        return str(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    @staticmethod
    def repack_args(cls: type, state: Dict) -> 'MetaDict':
        """Repack and rename keyword arguments stored in state before feeding to class constructor"""
        _data = state.pop('_data')
        del state['_memory_map']
        _nested_assignment = state.pop('_nested_assignment')
        return cls(_data, nested_assignment=_nested_assignment, **state)

    def __reduce__(self) -> Tuple:
        """Return state information for pickling."""
        return MetaDict.repack_args, (self.__class__, self.__dict__)

    def __dir__(self) -> Iterable[str]:
        """Extend dir list with accessible dict keys (enables autocompletion when using dot notation)"""
        dict_keys = [key for key in self._data.keys() if complies_variable_syntax(key)]
        return dir(type(self)) + dict_keys

    def copy(self) -> 'MetaDict':
        return self.__copy__()

    def __copy__(self) -> 'MetaDict':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    @classmethod
    def fromkeys(cls, iterable: Iterable[KT], value: Optional[VT] = None) -> 'MetaDict':
        return cls({key: value for key in iterable})

    def to_dict(self) -> Dict:
        return MetaDict.to_object(self._data)

    @staticmethod
    def to_object(obj: Any, _memory_map: Optional[Dict] = None) -> Any:
        if _memory_map is None:
            _memory_map = {}

        if id(obj) in _memory_map:
            return _memory_map[id(obj)]

        if isinstance(obj, (list, tuple, set)):
            if MetaDict._contains_mapping(obj):
                value = type(obj)(
                    MetaDict.to_object(x, _memory_map) if id(x) not in _memory_map else _memory_map[id(x)]
                    for x in obj
                )
            else:
                value = obj
        elif isinstance(obj, Mapping):
            value = {k: MetaDict.to_object(v, _memory_map) for k, v in obj.items()}
        else:
            value = obj

        _memory_map[id(obj)] = value

        return value

    def from_object(self, obj: Any) -> Any:

        if id(obj) in self._memory_map:
            return self._memory_map[id(obj)]

        if isinstance(obj, (list, tuple, set)):
            if MetaDict._contains_mapping(obj):
                value = type(obj)(self.from_object(x) if id(x) not in self._memory_map else self._memory_map[id(x)]
                                  for x in obj)
            else:
                value = obj
        elif isinstance(obj, MetaDict):
            value = obj
        elif isinstance(obj, Mapping):
            value = self.__class__({k: self.from_object(v) for k, v in obj.items()},
                                   nested_assignment=self._nested_assignment)
        else:
            value = obj

        self._memory_map[id(obj)] = value

        return value

    def _set_nested_assignment(self, val: bool):
        self.__dict__['_nested_assignment'] = val
        for key, value in self.items():
            if isinstance(value, (list, tuple, set)):
                for elem in value:
                    if isinstance(elem, MetaDict):
                        elem._set_nested_assignment(val)
            elif isinstance(value, MetaDict):
                value._set_nested_assignment(val)

    def enable_nested_assignment(self):
        self._set_nested_assignment(True)

    def disable_nested_assignment(self):
        self._set_nested_assignment(False)

    @contextlib.contextmanager
    def enabling_nested_assignment(self):
        """Context manager which temporarily enables nested key/attribute assignment."""
        nested_assignment = self.nested_assignment
        if not nested_assignment:
            self.enable_nested_assignment()
        try:
            yield self
        finally:
            if not nested_assignment:
                self.disable_nested_assignment()

    @property
    def nested_assignment(self):
        return self._nested_assignment

    @staticmethod
    def _contains_mapping(iterable: Iterable, ignore: Optional[type] = None) -> bool:
        """Recursively checks whether an Iterable contains an instance of Mapping."""
        for x in iterable:
            if isinstance(x, Mapping):
                if ignore is None or not isinstance(x, ignore):
                    return True
            elif isinstance(x, (list, set, tuple)):
                return MetaDict._contains_mapping(x, ignore)
        return False
