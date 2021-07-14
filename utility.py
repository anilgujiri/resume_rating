# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 06:12:41 2021

@author: Admin
"""

import pandas as pd
import tika
import pickle
import spacy
import docx
import re
import nltk
import sys


from nltk.corpus import stopwords
from spacy.language import Language

class utils: 
    class utilsError(Exception):
        pass

    def __init__(self):
        print("Loading nlp tools...")
        self.nlp = utils.loadDefaultNLP()
        print("Loading pdf parser...")
        tika.initVM()
        from tika import parser
        self.parser = parser


    
    def loadDefaultNLP(is_big = False):
        """
        Function to load the default SpaCy nlp model into nlp
        :Input is_big: if True, uses a large vocab set, else a small one
        :returns: nlp: a SpaCy nlp model
        """
    
        @Language.component("segment_on_newline")
        def segment_on_newline(doc):
            for token in doc[:-1]:
                if token.text.endswith("\n"):
                    doc[token.i + 1].is_sent_start = True
            return doc
    
        if is_big:
            nlp = spacy.load("en_core_web_lg")
        else:
            nlp = spacy.load("en_core_web_sm")
    
        Language.component("segment_on_newline",func=segment_on_newline)
        nlp.add_pipe("segment_on_newline",before="parser")
        return nlp
    
    

    def getDocxText(filename):
        """
        Get the text from a docx file
        :param filename: docx file
        :returns fullText: text of file
        """
        doc = docx.Document(filename)
        fullText = []
        for para in doc.paragraphs:
            txt = para.text
            fullText.append(txt)
        return "\n".join(fullText)
    
    
    
    def getPDFText(filename, parser):
        """
        Get the text from a pdf file
        :param filename: pdf file
        :param parser: pdf parser
        :returns fullText: text of file
        """
        raw = parser.from_file(filename)
        new_text = raw["content"]
        if "title" in raw["metadata"]:
            title = raw["metadata"]["title"]
            new_text = new_text.replace(title, "")
            
        return new_text
    
    
    def countWords(line):
        """
        Counts the numbers of words in a line
        :param line: line to count
        :return count: num of lines
        """
        count = 0
        is_space = False
        for c in line:
            is_not_char = not c.isspace()
            if is_space and is_not_char:
                count += 1
            is_space = not is_not_char
        return count

    def loadDocumentIntoSpacy(f, parser, spacy_nlp):
        """
        Convert file into spacy Document
        :param f: filename
        :param parser: pdf_parser
        :param spacy_nlp: nlp model
        :returns nlp_doc: nlp doc
        :returns new_text: text of file
        """
        if f.endswith(".pdf"):
            new_text = utils.getPDFText(f, parser)
        elif f.endswith(".docx"):
            new_text = utils.getDocxText(f)
        else:
            return None, None
    
        # new_text = "\n".join(
        #     [line.strip() for line in new_text.split("\n") if len(line) > 1]
        # )
    
        new_text = re.sub("\n{3,}", "\n", new_text)
        new_text = str(bytes(new_text, "utf-8").replace(b"\xe2\x80\x93", b""), "utf-8")
        # convert to spacy doc
        return spacy_nlp(new_text),new_text

            
    def getAllTokensAndChunks(doc):
        """
        Converts a spacy doc into tokens and chunks
        :Input doc: a SpaCy doc
        :returns: seen_chunks_words: set of strings seen
        :returns: all_tokens_chunks: set of all tokens and chunks found
        """
        # used to test duplicate words/chunks
        seen_chunks_words = set()
        # collect all words/chunks
        all_tokens_chunks = set()
        # generate all 1-gram tokens
        for token in doc:
            w = token.lemma_.lower()
            if (w not in seen_chunks_words):
                all_tokens_chunks.add(token)
                seen_chunks_words.add(w)
    
        # generate all n-gram tokens
        #print("\n Doc noun is ",doc.noun_chunks)
        for chunk in doc.noun_chunks:
            c = chunk.lemma_.lower()
            #print("\n Smaller chunk is", c)
            if (
                len(chunk) > 1
                and (c not in seen_chunks_words)
            ):
                all_tokens_chunks.add(chunk)
                seen_chunks_words.add(c)
    
        return seen_chunks_words, all_tokens_chunks

    

    def test(self,filename,info_extractor):
        """
        Test a document and print the extracted information and rating
        :param filename: name of resume file
        :param info_extractor: InfoExtractor object
        """
        
        doc, _ = utils.loadDocumentIntoSpacy(filename, self.parser, self.nlp)
        seen_chunks_words, all_tokens_chunks = utils.getAllTokensAndChunks(doc)
        seen_chunks_words.update(all_tokens_chunks)
        review = re.sub('http[s]?://\S+', '', str(seen_chunks_words))
        review = review.lower()
        review = re.sub('[^a-zA-Z&.]', ' ', review)
        print("Decoded the Doc")
        
        skills = []
        with open("skills.txt.txt",'r') as f:
           # perform file operations
            for word in f.readline().strip().split(','):
                word=re.sub('[^a-zA-Z]', ' ', str(word))
                skills.append(word)
                
        skills_updated=[]
        for word in skills:
            new_words=word.split()
            skills_updated.append(new_words)
        skills = []
        
        final_word = ""
        for word in skills_updated:
            concat_word = ""
            for j in word:
                if j in review:
                    concat_word = concat_word + j  
                    
            if concat_word != "":
                final_word = final_word +" "+concat_word
        
        print("Skills Fetched")        
        
        final_word =' '.join(map(str,[i for i in list(set(final_word.split(' '))) if len(i) > 2]))
        skills = [line.strip() for line in open("intermediate.txt", 'r')]
                
        final_word = nltk.word_tokenize(final_word)
        final_word = [word for word in final_word if not word in (set(stopwords.words('english')),set(skills))]
        
        print("Final Resume")
        
        df = pd.read_csv('columns.csv')
        df.loc[0] = 0
        for word in final_word:
            df[word] = 1
    
        
        df['Category_' + sys.argv[3]] = 1
        try:
            with open("finalized_model_cv.sav", 'rb') as file:  
                    Pickled_Model = pickle.load(file)
        except Exception as e:
            print(e)
            raise utils.utilsError(
                "File is not present in the path"
            )
        pred = Pickled_Model.predict(df)
        
        if info_extractor is not None:
            print("-" * 10)
            info_extractor.extractFromFile(filename)
            print("-" * 10)
        print("Rating: {}".format(pred) )
        return pred
    
        

