import ftfy
import re
from gensim.models.phrases import Phrases
from gensim.models.doc2vec import TaggedDocument
from string import digits, punctuation
import numpy as np


# Create a function to prepare documents for Doc2Vec and also make an iterable for Doc2Vec training
class DocIterator(object):
    # Initialize
    def __init__(self, uncleanDocList, labelList):
        # Set list of labels
        self.labelList = labelList
        # Set list of documents
        self.uncleanDocList = uncleanDocList

    # Create functions to ensure labels are unique
    def checkLabelsUnique(self):
        assert len(set(self.labelList)) == len(self.labelList), 'Labels are not unique!'

    # Create function to prepare documents by cleaning and optional phrase detection
    def prepareDocs(self, phrases=1):
        preppedDocs = []
        # Clean
        for i, doc in enumerate(self.uncleanDocList):
            cleanedDoc = ftfy.fix_text(doc, normalization='NFKC')
            cleanedDoc = cleanedDoc.replace('?', ' ')
            cleanedDoc = ' '.join(cleanedDoc.splitlines())
            cleanedDoc = re.sub(r'http\S+', '', cleanedDoc)
            cleanedDoc = re.sub(r'https\S+', '', cleanedDoc)
            translator = str.maketrans(punctuation, ' ' * len(punctuation))
            cleanedDoc = cleanedDoc.translate(translator)
            cleanedDoc = cleanedDoc.translate({ord(k): None for k in digits})
            cleanedDoc = cleanedDoc.lower()
            cleanedDoc = ' '.join(cleanedDoc.split())
            preppedDocs.append(cleanedDoc)
            print('%d of %d documents cleaned.' % (i + 1, len(self.uncleanDocList)))
        # Detect phrases (optional)
        if phrases is not None:
            print('Phrase detection requested. Running...')
            tokenizedDocs = []
            for doc in preppedDocs:
                tokenizedDocs.append(doc.split())
            bigrammer = Phrases(tokenizedDocs)
            preppedDocs = []
            for tokdoc in tokenizedDocs:
                preppedDocs.append(' '.join(bigrammer[tokdoc]))
            print('Documents are now phrase-collocated.')
        # Save prepared documents to class instance
        self.preppedDocList = preppedDocs

    # Create a function to make this class instance an iterable of named tuples, as required by Doc2Vec
    def __iter__(self):
        for i, doc in enumerate(self.preppedDocList):
            yield TaggedDocument(words=doc.split(), tags=[self.labelList[i]])

    # Create a function to shuffle documents, for better training
    def shuffleDocs(self):
        perm = np.random.permutation(len(self.preppedDocList))
        self.preppedDocList = list(np.array(self.preppedDocList)[perm])
        self.labelList = list(np.array(self.labelList)[perm])
