import nltk
import glob
import sys
import os
import bpe

def syllable_tokenize():
    SSP = nltk.tokenize.SyllableTokenizer()
    def tokenize(x):
        tkns = word_space_tokenize(x)
        syls = [syl for tkn in tkns for syl in SSP.tokenize(tkn)]

        return syls

    return tokenize

def bpe_tokenize(data_regex):
    all_files = []
    for file in glob.glob(data_regex):
        with open(file) as stream:
            all_files.append(stream.read())
    bpe_encoder = bpe.Encoder(ngram_max=5, pct_bpe=0.6, required_tokens=[" "])
    bpe_encoder.fit(all_files)

    def tokenize(x):
        tkns = bpe_encoder.tokenize(x)
        return [tkn for tkn in tkns if tkn not in ["__sow", "__eow"]]

    return tokenize


def word_space_tokenize(x):
    space_split = x.split(" ")
    return [tkn for word in space_split for tkn in nltk.word_tokenize(word) + [" "]]


if __name__ == "__main__":

    tokenize_method = sys.argv[1]
    data_regex = sys.argv[2]
    nltk.download('punkt')

    tokenize = lambda x: x.split(" ")
    if tokenize_method == "syllable":
        tokenize = syllable_tokenize()
    elif tokenize_method == "wordspace":
        tokenize = word_space_tokenize
    elif tokenize_method == "word":
        tokenize = nltk.word_tokenize
    elif tokenize_method == "bpe":
        tokenize = bpe_tokenize(data_regex)
        
    os.makedirs(tokenize_method, exist_ok=True)
    for file in glob.glob(data_regex):
        dst_file = os.path.join(tokenize_method, os.path.basename(file))
        with open(file) as datafile:
            tkns = tokenize(datafile.read())
        with open(dst_file, "w") as dstfile:
            dstfile.write("\n".join(tkns))

    
        