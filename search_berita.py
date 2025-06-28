# app.py
from flask import Flask, request, render_template
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import nltk, pandas as pd
import re
from collections import Counter

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

app = Flask(__name__)

doc_ids = []
tokenized_docs = []
doc_titles = []
doc_tags = []
doc_urls = []
doc_authors = []
doc_times = []
doc_years = []
available_years = []
bm25 = None
all_tags = []
tag_counts = {}

def normalize_tag(tag):
    tag = tag.strip().lower()
    tag = re.sub(r"[\"'`’‘”“\[\]]", "", tag)
    tag = re.sub(r"\s+", " ", tag)
    return tag.strip()


@app.before_request
def load_data():
    global doc_ids, tokenized_docs, doc_titles, doc_tags, doc_urls, doc_authors, doc_times, doc_years, available_years, bm25, all_tags, tag_counts, filter_author
    df = pd.read_csv("news_data.csv", usecols=['title','article_text','tag','url', 'author', 'publish_date'], nrows=5000)

    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')

    doc_ids = df.index.astype(str).tolist()
    doc_titles = df['title'].tolist()
    doc_urls = df['url'].tolist()
    doc_authors = df['author'].tolist()
    doc_times = df['publish_date'].dt.strftime('%Y-%m-%d').tolist()

    df['year'] = df['publish_date'].dt.year
    doc_years = df['year'].tolist()
    available_years = sorted(df['year'].dropna().unique().astype(int), reverse=True)


    tokenized_docs = []
    for title, text in zip(df['title'], df['article_text']):
        words = (str(title) + " " + str(text)).split()
        filtered = [w for w in words if w.lower() not in stop_words]
        tokenized_docs.append(filtered)

    bm25 = BM25Okapi(tokenized_docs)

    tag_set = set()
    cleaned_doc_tags = []
    tag_counter = Counter()

    for tags in df['tag'].fillna(''):
        cleaned_tags = []
        for tag in tags.split(','):
            cleaned = normalize_tag(tag)
            if cleaned:
                cleaned_tags.append(cleaned)
                tag_counter[cleaned] += 1
        cleaned_doc_tags.append(', '.join(cleaned_tags))
        tag_set.update(cleaned_tags)


    fully_cleaned_tags = {normalize_tag(t) for t in tag_set}

    doc_tags = cleaned_doc_tags
    all_tags = sorted(fully_cleaned_tags)
    tag_counts = dict(tag_counter)
    filter_author = [re.sub(r"[\"'`’‘”“\[\]]", "", author) for author in doc_authors]
    available_years = sorted(df['year'].dropna().unique().astype(int), reverse=True)


# Ganti seluruh fungsi index() Anda dengan yang ini
@app.route('/', methods=['GET', 'POST'])
def index():
    query = request.form.get('query', '').strip().lower()
    selected_tags = request.form.getlist('tag_filter')
    # selected_year = request.form.get('year_filter', '')

    date_from = request.form.get('date_from', '')
    date_to = request.form.get('date_to', '')


    selected_tags = [tag.lower() for tag in selected_tags]
    results = []

    if query:
        tokens = [w for w in query.split() if w.lower() not in stop_words]
        scores = bm25.get_scores(tokens)

        # Urutkan berdasarkan skor tertinggi
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        for idx, score in ranked:
            tags = doc_tags[idx].lower()
            # year = str(doc_times[idx])[:4]
            doc_date = str(doc_times[idx])[:10]  # pastikan YYYY-MM-DD format

            # Filter berdasarkan tag
            if selected_tags and not any(tag in tags for tag in selected_tags):
                continue

            # Filter berdasarkan tanggal range
            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue

            results.append({
                'title': doc_titles[idx],
                'tags': doc_tags[idx],
                'author': doc_authors[idx],
                'date': doc_times[idx],
                'score': score,
                'url': doc_urls[idx]
            })

    else:
        # Query kosong → hanya filter
        for idx in range(len(doc_titles)):
            tags = doc_tags[idx].lower()
            # year = str(doc_times[idx])[:4]
            doc_date = str(doc_times[idx])[:10]

            if selected_tags and not any(tag in tags for tag in selected_tags):
                continue

            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue

            results.append({
                'title': doc_titles[idx],
                'tags': doc_tags[idx],
                'author': doc_authors[idx],
                'date': doc_times[idx],
                'score': 0,
                'url': doc_urls[idx]
            })

    return render_template('index.html',
                            results=results,
                            all_tags=all_tags,
                            selected_tags=selected_tags,
                            tag_counts=tag_counts,
                            available_years=available_years,


                            date_from=date_from,
                            date_to=date_to,

                            query=query)


app.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

if __name__ == '__main__':
    app.run(debug=True)