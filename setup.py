from pathlib import Path

import setuptools

basedir = Path(__file__).resolve().parent

setuptools.setup(
    name='ucs',
    version='0.1',
    description='Implements the CAM02-UCS forward transform symbolically, using Theano.',
    long_description=(basedir/'README.rst').read_text(),
    url='https://github.com/crowsonkb/ucs',
    download_url='https://github.com/crowsonkb/ucs/archive/0.1.zip',
    author='Katherine Crowson',
    author_email='crowsonkb@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering',
    ],
    keywords='ciecam02 cam02 cam02-ucs color science',
    packages=['ucs'],
    install_requires=(basedir/'requirements.txt').read_text().split('\n'),
)
