import time

_this_year = time.strftime("%Y")
__version__ = '0.1.0'
__author__ = 'Lars Hillebrand'
__author_email__ = 'hokage555@web.de'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2022-{_this_year}, {__author__}.'
__homepage__ = 'https://github.com/LarsHill/metadict/'
__docs__ = (
    "MetaDict is a powerful dict subclass which enables dot-syntax, auto-completion "
    "and nested key access and assignment."
)

__all__ = ["__author__", "__author_email__", "__copyright__", "__docs__", "__homepage__", "__license__", "__version__"]
