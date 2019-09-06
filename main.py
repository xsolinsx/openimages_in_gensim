# garbage collector, used to free resources
import gc
# importlib.reload call is useful if you want to modify the parameters of the functions inside utilities.py at runtime
# # example
# create modelA with X and Y parameters set to 1, then update utilities.py with X and Y parameters set to 0, SAVE the file and create modelB
import importlib
import logging
import os
import random

import gensim
import pandas

import utilities

# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)

# initialize basic variables
corpus_path: str = None
model: gensim.models.word2vec.Word2Vec = None
model_path: str = None

# cycle through this forever (well, actually until "0" or "e" are chosen)
while True:
    print("""TOOL TO CONVERT OPENIMAGESv4 DATASETS (imagelables or bbox) INTO TEXT CORPORA AND MANAGE WORD2VEC MODELS POWERED BY GENSIM
0. Exit (e)
1. Create Corpus from OpenImages Dataset (cc)
2. Load Corpus (lc)
3. Get Word X Count (wc) requires 2
4. Create Model (cm) requires 2
5. Load Model (lm)
6. Train Model (tm) requires 2 and 5
7. Get Word Vector (wv) requires 5
8. Get Most Similar N Words to + and - ones using 'Cosine Distance' (ms) requires 5
9. Get Most Similar N Words to + and - ones using 'Multiplicative Combination' (msc) requires 5
10. Get Cosine Similarity between 2 Sets of Words (ns) requires 5
11. Get Most Similar N Words to Word (sw) requires 5
12. Get Cosine Similarity To Word (s) requires 5
13. Show Model's Graph (g) requires 5
14. Get Loaded Corpus' and Model's Paths (p)
""")
    function = input("Function: ").lower()
    # just a separator
    print("\n")
    if function == "e" \
            or function == "0":
        print("Press Enter to close.")
        input()
        # exit the "while True:"
        break
    elif function == "1" \
            or function == "cc":
        corpus_name = input("Corpus name: ")
        # use the path of main.py as directory
        corpus_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   corpus_name)

        dataset_path = input(
            "Dataset ABSOLUTE path (leave empty if it is in the same folder as this tool and is named 'training.csv'): ")
        # use the path of main.py as directory if empty input
        dataset_path = os.path.join(dataset_path if dataset_path else os.path.dirname(os.path.abspath(__file__)),
                                    "training.csv")

        labels_path = input(
            "Labels ABSOLUTE path (leave empty if it is in the same folder as this tool and is named 'labels.csv'): ")
        # use the path of main.py as directory if empty input
        labels_path = os.path.join(labels_path if labels_path else os.path.dirname(os.path.abspath(__file__)),
                                   "labels.csv")

        confidence = input(
            "Confidence (default 0.85 USE DOT AS DECIMAL SEPARATOR): ")
        # if input convert to float else default
        confidence = float(confidence) if confidence else 0.85

        print()
        print("I'm loading labels. {0}".format(labels_path))
        # read class descriptions from file
        labels = pandas.read_csv(labels_path,
                                 header=None)

        print("I'm processing labels (everything lowercase and underscores replacing white spaces). ")
        # see function inside utilities.py
        labels[1] = labels[1].apply(utilities.adjust_labels)

        print("I'm loading the dataset in memory. {0}".format(dataset_path))
        # read training file
        dataset = pandas.read_csv(dataset_path,
                                  header=0)

        print("I'm merging labels and the dataset assuming the first column on labels.csv is the key matching with LabelName on training.csv.")
        # merge the two tables over the labels' ids
        data = pandas.merge(left=labels,
                            right=dataset,
                            how="inner",
                            left_on=0,
                            right_on="LabelName")
        # now data is LabelName(0),Description(1),ImageID(2),Source(3),LabelName(4),Confidence(5),XMin(6),XMax(7),YMin(8),YMax(9),IsOccluded(10),IsTruncated(11),IsGroupOf(12),IsDepiction(13),IsInside(14)
        # columns from 6 to 14 are not present if you use imagelabels dataset
        # access with indexes
        print("I'm freeing up some RAM (labels and dataset).")
        del labels
        del dataset
        gc.collect()

        print(
            "I'm processing the corpus (keeping every object with confidence >= {0} for each image).".format(confidence))
        corpus: dict = dict()
        # for each row(object in image) append/ignore the object's label while adding the id of the image to the corpus dictionary as key (if not already present)
        for row in data.values:
            if float(row[5]) >= confidence:
                if row[2] not in corpus:
                    corpus[row[2]] = list()
                corpus[row[2]].append(row[1])
        print("I'm freeing up some RAM (data).")
        del data
        gc.collect()

        print("I'm writing the corpus randomizing the order of sentences(images) and words(objects) inside them. " + corpus_name)
        # save keys inside a list and shuffle them
        keys = list(corpus.keys())
        random.shuffle(keys)
        # write corpus on file in order to not load it again in RAM
        with open(file=corpus_name,
                  mode="w",
                  encoding="utf-8") as f:
            # for each image_id retrieve the objects from the corpus and shuffle them
            for image_id in keys:
                objects = corpus[image_id]
                if objects:
                    # if image is empty(objects with low confidence) ignore
                    random.shuffle(objects)
                    f.write(" ".join(objects) + "\n")
        print("I'm freeing up some RAM (file and corpus).")
        del corpus
        gc.collect()
    elif function == "2" \
            or function == "lc":
        corpus_path = None
        corpus_path = input("Corpus ABSOLUTE path: ")
        if os.path.isfile(corpus_path):
            print("I'm loading the corpus. {0}".format(corpus_path))
            with open(file=corpus_path,
                      mode="r",
                      encoding="utf-8") as f:
                corpus: list = f.readlines()
                print("Corpus loaded, I'm computing some stats.")
                counter = list()
                for sentence in corpus:
                    counter.append(sentence.count(" ") + 1)
                print("Total sentences: {0}".format(len(corpus)))
                print("Total objects: {0}".format(sum(counter)))
                print("Average objects per image: {0}\nMax: {1}\nMin: {2}".format(sum(counter) / len(counter),
                                                                                  max(counter),
                                                                                  min(counter)))

                print("I'm freeing up some RAM (file, corpus and counter).")
                del corpus
                del counter
            gc.collect()
        else:
            print("Corpus file not found. {0}".format(corpus_path))
    elif function == "3" \
            or function == "wc":
        print(
            "THIS FUNCTION EXCLUDES ALL IMAGES IN WHICH THE SPECIFIED WORD IS NOT PRESENT.")

        target = input("Insert the desired word: ").lower()

        print("I'm loading the corpus. {0}".format(corpus_path))
        with open(file=corpus_path,
                  mode="r",
                  encoding="utf-8") as f:
            corpus: list = f.readlines()
            print("Corpus loaded, I'm computing some stats.")
            counter = dict()
            i = 0
            # count occurrencies of a word and images in which it appears
            for sentence in corpus:
                for word in sentence.split(" "):
                    # replace newline if present
                    if word.replace("\n", "") == target:
                        if i not in counter:
                            counter[i] = 0
                        counter[i] += 1
                i += 1
            if counter:
                print("Total occurrencies: {0}".format(sum(counter.values())))
                print("Total images: {0}".format(len(counter)))
                print("Average occurrencies per image: {0}\nMax: {1}\nMin: {2}".format(sum(counter.values()) / (len(counter) if len(counter) > 0 else 1),
                                                                                       max(counter.values(
                                                                                       )),
                                                                                       min(counter.values())))
            else:
                print("{0} not in the corpus".format(target))

            print("I'm freeing up some RAM (file, corpus and counter).")
            del corpus
            del counter
        gc.collect()
    elif function == "4" \
            or function == "cm":
        model_name = input("Model name: ")

        vocabulary_name = input("Vocabulary name (leave empty if you don't want to save the vocabulary): ")

        # see first lines of file for explanation of reload call
        importlib.reload(utilities)
        # see function inside utilities.py for parameters (instantiation of the model)
        utilities.create_model(corpus_path=corpus_path,
                               model_name=model_name,
                               vocabulary_name=vocabulary_name)
    elif function == "5" \
            or function == "lm":
        model_path = None
        model = None
        model_path = input("Model ABSOLUTE path: ")
        if os.path.isfile(model_path):
            is_full_model = input("Is it a full model or just word vectors? (1 for full model, 0 for word vectors): ")
            is_full_model = bool(int(is_full_model))
            if is_full_model:
                print("I'm loading the model. {0}".format(model_path))

                model = gensim.models.word2vec.Word2Vec.load(model_path)
            else:
                is_binary = input("Is it in the binary format? (1 for True, 0 for False): ")
                is_binary = bool(int(is_binary))

                print("I'm loading the model. {0}".format(model_path))

                model = gensim.models.KeyedVectors.load_word2vec_format(model_path,
                                                                        binary=is_binary)

            print("Model loaded.")

            print("Length of the vocabulary: {0}".format(len(model.wv.vocab.keys())))
        else:
            print("Model file not found. {0}".format(model_path))
    elif function == "6" \
            or function == "tm":
        new_model_name = input("New model name: ")

        new_vocabulary_name = input(
            "New vocabulary name (leave empty if you don't want to save the vocabulary): ")

        # see first lines of file for explanation of reload call
        importlib.reload(utilities)
        # see function inside utilities.py for parameters (model.train call)
        utilities.train_model(model=model,
                              corpus_path=corpus_path,
                              new_model_name=new_model_name,
                              new_vocabulary_name=new_vocabulary_name)
    elif function == "7" \
            or function == "wv":
        word = input("Insert the word of which you want the vector: ").lower()
        if word in model.wv.vocab.keys():
            print("This is the vector of {0}. ".format(word))
            print(model.wv.word_vec(word=word))
        else:
            print("Word not in the vocabulary of this model.")
    elif function == "8" \
            or function == "ms":
        print("WORDS MUST BE SEPARATED BY WHITE SPACES, IF YOU HAVE 'italian food' MAKE IT 'italian_food'.")
        print("YOU MUST SPECIFY AT LEAST ONE BETWEEN 'positive' AND 'negative' WORDS.")

        positive = input("Insert the words which contribute positively: ")
        # if input then lowercase all words and split them by " " else empty list
        positive = positive.lower().split(" ") if positive else list()

        negative = input("Insert the words which contribute negatively: ")
        # if input then lowercase all words and split them by " " else empty list
        negative = negative.lower().split(" ") if negative else list()

        topn = input("Insert the number of words you want (default 10): ")
        # if input convert to int else default
        topn = int(topn) if topn else 10

        # if all the words specified are in the dicitonary then process
        if set(positive + negative).issubset(model.wv.vocab.keys()):
            print("I'm computing the result.")
            # join and separate with spaces every item of the array returned by the function on a string
            print("The result is: {0}".format(" ".join([str(x) for x in model.wv.most_similar(positive=positive,
                                                                                              negative=negative,
                                                                                              topn=topn)])))
        else:
            # \a is beep
            print("\aERROR")

            missing = list()
            # find missing word(s)
            for word in positive + negative:
                if word not in model.wv.vocab.keys():
                    missing.append(word)
            print("Some words are not in the vocabulary: {0}".format(
                " ".join(missing)))
    elif function == "9" \
            or function == "msc":
        print("WORDS MUST BE SEPARATED BY WHITE SPACES, IF YOU HAVE 'italian food' MAKE IT 'italian_food'.")
        print("YOU MUST SPECIFY AT LEAST ONE BETWEEN 'positive' AND 'negative' WORDS.")

        positive = input("Insert the words which contribute positively: ")
        # if input then lowercase all words and split them by " " else empty list
        positive = positive.lower().split(" ") if positive else list()

        negative = input("Insert the words which contribute negatively: ")
        # if input then lowercase all words and split them by " " else empty list
        negative = negative.lower().split(" ") if negative else list()

        topn = input("Insert the number of words you want (default 10): ")
        # if input convert to int else default
        topn = int(topn) if topn else 10
        # if all the words specified are in the dicitonary then process
        if set(positive + negative).issubset(model.wv.vocab.keys()):
            print("I'm computing the result.")
            # join and separate with spaces every item of the array returned by the function on a string
            print("The result is: {0}".format(" ".join([str(x) for x in model.wv.most_similar_cosmul(positive=positive,
                                                                                                     negative=negative,
                                                                                                     topn=topn)])))
        else:
            # \a is beep
            print("\aERROR")

            missing = list()
            # find missing word(s)
            for word in positive + negative:
                if word not in model.wv.vocab.keys():
                    missing.append(word)
            print("Some words are not in the vocabulary: {0}".format(
                " ".join(missing)))
    elif function == "10" \
            or function == "ns":
        print("WORDS MUST BE SEPARATED BY WHITE SPACES, IF YOU HAVE 'italian food' MAKE IT 'italian_food'.")

        ws1 = input("Insert the words of set 1: ")
        # if input then lowercase all words and split them by " " else empty list
        ws1 = ws1.lower().split(" ") if ws1 else list()

        ws2 = input("Insert the words of set 2: ")
        # if input then lowercase all words and split them by " " else empty list
        ws2 = ws2.lower().split(" ") if ws2 else list()

        # if all the words specified are in the dicitonary then process
        if set(ws1 + ws2).issubset(model.wv.vocab.keys()):
            print("I'm computing the result.")
            print("The result is: {0}".format(model.wv.n_similarity(ws1=ws1,
                                                                    ws2=ws2)))
        else:
            # \a is beep
            print("\aERROR")

            missing = list()
            # find missing word(s)
            for word in ws1 + ws2:
                if word not in model.wv.vocab.keys():
                    missing.append(word)
            print("Some words are not in the vocabulary: {0}".format(
                " ".join(missing)))
    elif function == "11" \
            or function == "sw":
        print("IF YOU HAVE 'italian food' MAKE IT 'italian_food'.")

        word = input("Insert the desired word: ").lower()

        topn = input("Insert the number of words you want (default 10): ")
        # if input convert to int else default
        topn = int(topn) if topn else 10

        if word in model.wv.vocab.keys():
            print("I'm computing the result.")
            # join and separate with spaces every item of the array returned by the function on a string
            print("The result is: {0}".format(" ".join([str(x) for x in model.wv.similar_by_word(word=word,
                                                                                                 topn=topn)])))
        else:
            # \a is beep
            print("\aERROR")
            print("{0} is not in the vocabulary.".format(word))
    elif function == "12" \
            or function == "s":
        print("IF YOU HAVE 'italian food' MAKE IT 'italian_food'.")

        w1 = input("Insert word 1: ").lower()

        w2 = input("Insert word 2: ").lower()

        # if all the words specified are in the dicitonary then process
        if set([w1, w2]).issubset(model.wv.vocab.keys()):
            print("I'm computing the result.")
            print("The result is: {0}".format(model.wv.similarity(w1=w1,
                                                                  w2=w2)))
        else:
            # \a is beep
            print("\aERROR")
            print("{0} and/or {1} are/is not in the vocabulary.".format(w1,
                                                                        w2))
    elif function == "13" \
            or function == "g":
        # see first lines of file for explanation of reload call
        importlib.reload(utilities)
        # see function inside utilities.py for parameters (TSNE call)

        print("I'm computing the graph, for models with a big vocabulary this might require a lot.")
        utilities.tsne_plot(model=model)
    elif function == "14" \
            or function == "p":
        print("The loaded corpus is: {0}".format(corpus_path))

        print("The loaded model is: {0}".format(model_path if model else None))
    else:
        print("Invalid Choice. Press Enter to return to the Main Menu.")
        input()
        # skip the next two lines and restart the cycle
        continue

    print("Done. Press Enter to return to the Main Menu.")
    input()
    # just a separator
    print("\n" * 10)
