from distutils.core import setup

setup(
    name='OntologyAlignmentWithEmbeddings',
    version='1.0',
    packages=['OntAlignEmbeddings',],
    license='Apache 2.0 software license',
    long_description=open('README.md').read(),
    install_requires=['gensim',
    				'scipy',
    				'nltk',
    				'pandas',
    				'polyglot',
    				'numpy',
                    're']
)