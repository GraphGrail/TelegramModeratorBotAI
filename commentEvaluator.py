import os
import sys
import pickle
import settings
import spacy
from spacy.matcher import Matcher
import pymorphy2
from keras.preprocessing import sequence
import tensorflow as tf


class CommentEvaluator():
    def __init__(self, model, tok):
        self.model = model

        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        self.tok = tok
        self.nlp_ = spacy.load('xx_ent_wiki_sm')
        self.morph_ = pymorphy2.MorphAnalyzer()
        
        self.negativePatterns_ = []
        
        # Угроза бана
        self.negativePatterns_.append([{"LOWER": "бан"}])
        self.negativePatterns_.append([{"LOWER": "забанить"}])
        self.negativePatterns_.append([{"LOWER": "банить"}])
        
        # Критика команды
        self.negativePatterns_.append([{"LOWER": "причина"}, {"OP": "+"}, {"LOWER": "задержка"}])
        self.negativePatterns_.append([{"LOWER": "когда"}, {"OP": "+"}, {"LOWER": "отчёт"}])
        self.negativePatterns_.append([{"LOWER": "задерживать"}, {"OP": "+"}, {"LOWER": "отчёт"}])
        self.negativePatterns_.append([{"LOWER": "задержать"}, {"OP": "+"}, {"LOWER": "отчёт"}])
        
        self.negativePatterns_.append([{"LOWER": "когда"}, {"OP": "+"}, {"LOWER": "исправить"}])
        
        self.negativePatterns_.append([{"LOWER": "удалять"}, {"OP": "+"}, {"LOWER": "сообщение"}])
        
        # Мошенничество
        self.negativePatterns_.append([{"LOWER": "мошенничество"}])
        self.negativePatterns_.append([{"LOWER": "мошеннический"}])
        self.negativePatterns_.append([{"LOWER": "кидалово"}])
        self.negativePatterns_.append([{"LOWER": "кинуть"}])
        self.negativePatterns_.append([{"LOWER": "распил"}])
        self.negativePatterns_.append([{"LOWER": "развод"}])
        
        self.negativePatterns_.append([{"LOWER": "совесть"}, {"OP": "+"}, {"LOWER": "мучить"}])
        self.negativePatterns_.append([{"LOWER": "никакой"}, {"OP": "+"}, {"LOWER": "совесть"}])
        
        self.negativePatterns_.append([{"LOWER": "дарить"}, {"OP": "+"}, {"LOWER": "деньга"}])
        self.negativePatterns_.append([{"LOWER": "подарить"}, {"OP": "+"}, {"LOWER": "деньга"}])
        self.negativePatterns_.append([{"LOWER": "отдать"}, {"OP": "+"}, {"LOWER": "деньга"}])
        
        # Цена токена
        self.negativePatterns_.append([{"LOWER": "сливать"}, {"OP": "+"}, {"LOWER": "токен"}])
        self.negativePatterns_.append([{"LOWER": "токен"}, {"OP": "+"}, {"LOWER": "фантик"}])
        self.negativePatterns_.append([{"LOWER": "фантики-токены"}])
        self.negativePatterns_.append([{"LOWER": "цена"}, {"OP": "+"}, {"LOWER": "была"}, {"OP": "+"}, {"LOWER": "сейчас"}])
        self.negativePatterns_.append([{"LOWER": "цена"}, {"OP": "+"}, {"LOWER": "токена"}, {"OP": "+"}, {"LOWER": "ниже"}])
        self.negativePatterns_.append([{"LOWER": "цена"}, {"OP": "+"}, {"LOWER": "токена"}, {"OP": "+"}, {"LOWER": "низкая"}])
        
        # Критика технологии
        self.negativePatterns_.append([{"LOWER": "медленно"}, {"OP": "+"}, {"LOWER": "работает"}])
        self.negativePatterns_.append([{"LOWER": "подвисает"}])
        self.negativePatterns_.append([{"LOWER": "подтормаживает"}])
        self.negativePatterns_.append([{"LOWER": "тормозит"}])
        self.negativePatterns_.append([{"LOWER": "баг"}])
        self.negativePatterns_.append([{"LOWER": "баги"}])
        self.negativePatterns_.append([{"LOWER": "глюк"}])
        self.negativePatterns_.append([{"LOWER": "глючит"}])
        self.negativePatterns_.append([{"LOWER": "лагает"}])

        self.negativePhraseMatcher_ = Matcher(self.nlp_.vocab)
        
        i = 0
        while i < len(self.negativePatterns_):
            self.negativePhraseMatcher_.add(str(i), None, self.negativePatterns_[i])
            i += 1
        
        return
    
    def analyze(self, doc, use_only_ml=False):
        if not use_only_ml:
            tokens = self.nlp_(doc)
            lemmatized_word_list = self.lemmatize_tokens(tokens)
            lemmatized_tokens = self.nlp_(" ".join(lemmatized_word_list))

            matches = self.negativePhraseMatcher_(tokens)
            if len(matches) > 0:
                return "negative"

            matches = self.negativePhraseMatcher_(lemmatized_tokens)
            if len(matches) > 0:
                return "negative"

        check = sequence.pad_sequences(self.tok.texts_to_sequences([doc]), maxlen=settings.MAX_LENGTH)
        print('Message:', doc)
        with self.graph.as_default():
            answer = self.model.predict(check, batch_size=1024)
        print('Res:', answer)
        if answer[0][0] >= 0.17:
            return "negative"
        elif answer[0][0] > 0.04 and answer[0][0] < 0.17:
            return "neutral"
        else:
            return "positive"
        
    def lemmatize_tokens(self, tokens):
        res = []
        for t in tokens:
            res.append(self.morph_.parse(t.text)[0].normal_form)
        return res
    
# Fields:
    
    se_ = None
    nlp_ = None
    morph_ = None
    negativePhraseMatcher_ = None
    
    negativePatterns_ = None
