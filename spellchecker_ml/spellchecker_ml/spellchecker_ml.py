from spellchecker import SpellChecker as SpellChecker
from text_processing_ml.normalization import NormalizeText
import nltk
import numpy as np
import sys, json, codecs, pickle, random, time
from datetime import datetime
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm
import autocomplete

class SpellCheckerML:
    def __init__(self, label_encoder=None):
        self.label_encoder = label_encoder
        self._spell_checker = SpellChecker()
        
    def encode_text(self, text):
        lines = [line.split(' ') for line in text.split("\n")]
        words = []
        for line in lines:
            for word in line:
                words.append(word.lower())
        alphabet = set(words)
        if not self.label_encoder:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(list(alphabet))
        seq = self.label_encoder.transform(words)
        features = np.fromiter(seq, np.int64)
        features = np.atleast_2d(features).T
        lengths = [len(line) for line in lines]
        return features, lengths
    
    def train_hmm(self, text, num_states):
        np.random.seed(seed=42)
        features, lengths = self.encode_text(text)
        self.model = hmm.MultinomialHMM(n_components=num_states, init_params='ste')
        self.model.fit(features, lengths)

    def save_model(self):
        joblib.dump(self.model, "hmm_model.pkl")
        
    def save_label_encoder(self):
        with open("hmm_label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def load_hmm(self, label_encoder_path, hmm_model_path):
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        model = joblib.load(hmm_model_path)
        return label_encoder, model

    def predict_proba_hmm(self, word):
        features, lengths = self.encode_text(word)
        return model.predict_proba(features, lengths=lengths)
        
    def correction(self, word):
        candidates = self._spell_checker.candidates(word)
        probabilities = []
        print(candidates)
        for candidate in candidates:
            try:
                probabilities.append(
                    self.predict_proba_hmm(candidate)
                )
            except:
                continue
        if probabilities == []:
            raise Exception("""No replacement suggestions found. 
            Consider using a bigger corpus.""")
        index = np.argmax(probabilities)
        return candidates[index]
        
