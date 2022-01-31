import os
from importlib.util import module_from_spec, spec_from_file_location
from setuptools import setup, find_packages


_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(file_name: str, pkg="metadict"):
    spec = spec_from_file_location(os.path.join(pkg, file_name), os.path.join(_PATH_ROOT, pkg, file_name))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

test_deps = ['pytest']

setup(name='metadict',
      version=about.__version__,
      description=about.__docs__,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=about.__author__,
      author_email=about.__author_email__,
      url=about.__homepage__,
      download_url=about.__homepage__,
      license=about.__license__,
      copyright=about.__copyright__,
      keywords=['dict', 'attribute-style syntax', 'nesting', 'auto-completion'],
      packages=find_packages(),
      python_requires='>=3.6',
      extras_require={'tests': test_deps},
      tests_require=test_deps)
