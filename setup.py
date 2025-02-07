import os
from setuptools import setup
file_path = os.path.dirname(os.path.realpath(__file__))

try:
    version = os.environ['SIAC_VERSION']
except:
    version_file = open(os.path.join(file_path, 'SIAC/VERSION'), 'rb')
    version = version_file.read().decode().strip()

with open('README.md', 'rb') as f:
    readme = f.read().decode()

setup(name                          = 'SIAC',
      version                       = version,
      description                   = 'A sensor invariant Atmospheric Correction (SIAC)',
      long_description              = readme,
      long_description_content_type ='text/markdown',
      author                       = 'Feng Yin',
      author_email                 = 'ucfafyi@ucl.ac.uk',
      classifiers                  = ['Development Status :: 4 - Beta',
                                      'Programming Language :: Python :: 2.7',
                                      'Programming Language :: Python :: 3.6'],
      install_requires             = [],
      url                          = 'https://github.com/MarcYin/SIAC',
      license                      = "GNU Affero General Public License v3.0",
      include_package_data         = True,
      packages                     = ['SIAC'],
     )
