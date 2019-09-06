# garbage collector, used to free resources
import gc
import os

import gensim
from matplotlib import pyplot
from sklearn.manifold import TSNE


def adjust_labels(s: str):
    "Replace white spaces with underscores and make everything lowercase"
    return s.replace(" ", "_").lower()


def tsne_plot(model: gensim.models.word2vec.Word2Vec):
    "Creates and TSNE (t-distributed stochastic neighbor embedding) model and plots it"
    # source https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(n_components=2,
                      perplexity=40,
                      n_iter=2500,
                      init="pca",
                      verbose=1,
                      random_state=23)
    new_values = tsne_model.fit_transform(X=tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    pyplot.figure()
    for i in range(len(x)):
        pyplot.scatter(x=x[i],
                       y=y[i])
        pyplot.annotate(s=labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    pyplot.show()


def create_model(corpus_path: str,
                 model_name: str,
                 vocabulary_name: str = ""):
    """
    corpus_path : str
        Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
    model_name : str
        Name of the model file.
    vocabulary_name : str, optional
        Name of the vocabulary file.
    """
    print("I'm creating and training the model. " + model_name)
    # build the corpus as a vocabulary with the word2vec model, every word becomes an X dimensions array (size parameter) within a window of Y words to consider (window) excluding words that occur less than Z times (min_count)
    # the higher min_count, the more selective the model, the lower min_count, the less selective the model
    model = gensim.models.word2vec.Word2Vec(corpus_file=corpus_path,
                                            size=300,
                                            window=5,
                                            min_count=10,
                                            sample=1e-5,
                                            negative=3,
                                            sg=0,
                                            iter=5,
                                            workers=4)
    print("Model created.")

    print("I'm saving the model. " + model_name)
    # use the path of utilities.py as directory
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            model_name))

    if vocabulary_name:
        # Write out the vocabulary.
        print("I'm writing the vocabulary in alphabetical order. " + vocabulary_name)
        words = list(model.wv.vocab.keys())
        words.sort()
        # use the path of utilities.py as directory
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               vocabulary_name),
                  mode='w',
                  encoding="utf-8") as f:
            for word in words:
                f.write(word + '\n')
        print("Vocabulary written.")
        print("Length of the vocabulary: {0}".format(len(model.wv.vocab.keys())))
    else:
        print("Vocabulary skipped.")

    print("I'm freeing up some RAM (model).")
    del model
    gc.collect()


def train_model(model: gensim.models.word2vec.Word2Vec,
                corpus_path: str,
                new_model_name: str,
                new_vocabulary_name: str = ""):
    """
    model : :class:`~gensim.models.word2vec.Word2Vec`
        Word2Vec model.
    corpus_path : str
        Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
    new_model_name : str
        Name of the new model file.
    new_vocabulary_name : str, optional
        Name of the new vocabulary file.
    """
    print("I'm training the model with corpus. " + corpus_path)
    # INSERT MISSING PARAMETERS
    model.train(corpus_file=corpus_path)
    print("Model trained.")

    print("I'm saving the new model. " + new_model_name)
    # use the path of utilities.py as directory
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            new_model_name))

    if new_vocabulary_name:
        # Write out the vocabulary.
        print("I'm writing the new vocabulary in alphabetical order. " + new_vocabulary_name)
        words = list(model.wv.vocab.keys())
        words.sort()
        # use the path of utilities.py as directory
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               new_vocabulary_name),
                  mode='w',
                  encoding="utf-8") as f:
            for word in words:
                f.write(word + '\n')
        print("Vocabulary written.")
        print("Length of the vocabulary: {0}".format(len(model.wv.vocab.keys())))
    else:
        print("Vocabulary skipped.")

    print("I'm freeing up some RAM (model).")
    del model
    gc.collect()
