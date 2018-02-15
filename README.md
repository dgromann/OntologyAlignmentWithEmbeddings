# OntologyAlignmentWithEmbeddings

This project has been created for an LREC 2018 paper to align labels of two lightweight ontologies (industry classification systems) in four languages (en, de, it, es) using pre-trained embedding libraries. We were interested in seeing whether good results can be achieved with using already existing embeddings. It turned out that for our domain-specific multilingual scenario fastText provided the most successful results. For detailed results please consult our LREC 2018 paper. 


## Embedding repositories
The pretrained embeddings utilized in this project were retrieved from the following repositories.
* [word2vec](https://github.com/Kyubyong/wordvectors)
* [polyglot](https://sites.google.com/site/rmyeid/projects/polyglot)
* [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

## Dependencies
* gensim version 3.0.0
* numpy version 1.13.3
* nltk version 3.2.1
* pandas version 0.18.1
* polyglot version 16.07.04
* re version 2.2.1
* scipy version 0.19.1


## References 
If you use any of this code please cite the following paper: 
Gromann, D. and Declerck, T. "Comparing Pretrained Multilingual Word Embeddings on an Ontology Alignment Task", In: LREC 2018.