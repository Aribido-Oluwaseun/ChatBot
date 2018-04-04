from nltk.corpus import stopwords
import string
import math

class Similarity:

    def jaccard_similarity(self, query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection)/len(union)


    def remove_stopwords(self, tokenized_docs):
        filtered_words=[]
        filtered_words_docs=[]
        for i in range(len(tokenized_docs)):
            for j in range(len(tokenized_docs[i])):

                if tokenized_docs[i][j] not in stopwords.words('english'):
                    filtered_words.append(tokenized_docs[i][j])

            filtered_words_docs.append(list(filtered_words))
            del filtered_words[:]

        return filtered_words_docs


    def term_frequency(self, term, tokenized_docs):



def main():

    tokenize = lambda doc: doc.lower().split(" ")
    document_0 = "How to install RadioShop?"
    document_1 = "What is Technical Implementation Engineering?"
    document_2 = "How do you perform PCA testing?"
    document_3 = "What is PCA?"
    document_4 = "What is the difference between Mesh and Mesh IP?"
    document_5 = "What softwares do I need to install for working as a TIE?"
    document_6 = "What is SBS?"
    document_7 = "How do you birth chirp using RadioShop?"

    all_docs = [document_0, document_1, document_2, document_3, document_4, document_5, document_6, document_7]

    print(all_docs)

    tokenized_docs = [tokenize(d) for d in all_docs]  # tokenized docs
    all_tokens_set = set([item for sublist in tokenized_docs for item in sublist])

    print(all_tokens_set)
    query = "What is RadioShop?"

    tokenized_query = tokenize(query)

    sim = Similarity()

    print(tokenized_query)
    print(stopwords.words('english'))
    for i in range(len(tokenized_docs)):
        print(sim.jaccard_similarity(tokenized_query, tokenized_docs[i]))

    print(tokenized_docs)

    stop_words_removed_docs = sim.remove_stopwords(tokenized_docs)
    print(stop_words_removed_docs)
    print('\n')
    print('Output of Jaccard Similarity measure:')
    for i in range(len(stop_words_removed_docs)):
        print(sim.jaccard_similarity(tokenized_query, stop_words_removed_docs[i]))


if __name__ == "__main__":
    main()
