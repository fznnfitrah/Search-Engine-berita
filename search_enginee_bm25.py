from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import nltk, pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_corpus_from_csv(csv_file, n_rows=None):
    df = pd.read_csv(csv_file, usecols=['title','article_text','tag'], nrows=n_rows)

    doc_ids = df.index.astype(str).tolist()  # ID dokumen pakai index
    tokenized_docs = []
    doc_tags = []  # daftar tags untuk setiap dokumen

    for title, text, tags in zip(df['title'], df['article_text'], df['tag']):
        words = (str(title) + " " + str(text)).split()
        filtered = [w for w in words if w.lower() not in stop_words]
        tokenized_docs.append(filtered)
        doc_tags.append(tags)
    
    return doc_ids, tokenized_docs, doc_tags


def load_queries(query_file):
    # tetap baca file teks jika hanya ingin file query manual
    queries = []
    with open(query_file, encoding='utf-8') as f:
        for line in f:
            tokens = [w for w in line.split() if w.lower() not in stop_words]
            if tokens:
                queries.append(tokens)
    return queries

def search_bm25(doc_ids, tokenized_docs, doc_tags, queries, top_n=5):
    bm25 = BM25Okapi(tokenized_docs)
    for qid, query in enumerate(queries, start=1):
        scores = bm25.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        print("Queryid\tQ0\tDoc_id\tRank\tBM25_Score\tTags")
        for rank, (idx, score) in enumerate(ranked[:top_n], start=1):
            print(f"{qid}\tQ0\t{doc_ids[idx]}\t{rank}\t{score:.4f}\t{doc_tags[idx]}")


def main():
    corpus_csv = "news_data.csv"
    query_file = "queries.txt"
    doc_ids, tokenized_docs, doc_tags = load_corpus_from_csv(corpus_csv, n_rows=5000)
    queries = load_queries(query_file)
    search_bm25(doc_ids, tokenized_docs, doc_tags, queries, top_n=5)

if __name__ == "__main__":
    main()
