# app.py
from flask import Flask, request, render_template
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import nltk, pandas as pd
import re
from math import ceil
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
    global doc_ids, tokenized_docs, doc_titles, doc_tags, doc_urls, doc_authors, doc_times, doc_years, available_years, bm25, all_tags, tag_counts, all_author
    df = pd.read_csv("news_data.csv", usecols=['title','article_text','tag','url', 'author', 'publish_date'], nrows=5000)

    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')

    doc_ids = df.index.astype(str).tolist()
    doc_titles = df['title'].tolist()
    doc_urls = df['url'].tolist()
    doc_authors = [a if pd.notna(a) else '' for a in df['author'].tolist()]
    doc_times = df['publish_date'].dt.strftime('%Y-%m-%d').tolist()

    df['year'] = df['publish_date'].dt.year
    doc_years = df['year'].tolist()


    tokenized_docs = []
    for title, text in zip(df['title'], df['article_text']):
        if pd.isna(title) and pd.isna(text):
            continue  
        combined = f"{str(title)} {str(text)}"
        words = combined.split()
        filtered = [w for w in words if w.lower() not in stop_words]
        if filtered:  
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
    all_author = sorted(set(author.strip() for author in doc_authors if pd.notna(author)))
    available_years = sorted(df['year'].dropna().unique().astype(int), reverse=True)



from flask import request, render_template
from math import ceil

@app.route('/', methods=['GET'])
def index():
    # Ambil semua data dari request.args (karena pakai method=GET)
    query = request.args.get('query', '').strip().lower()
    selected_authors = [a.strip().lower() for a in request.args.getlist('author_filter')]
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    page = max(1, int(request.args.get("page", 1)))
    per_page = 10

    results = []

    if query:
        tokens = [w for w in query.split() if w.lower() not in stop_words]
        scores = bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        for idx, score in ranked:
            author = doc_authors[idx].strip().lower()
            doc_date = str(doc_times[idx])[:10]

            if selected_authors and author not in selected_authors:
                continue
            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue
            if score <= 0:
                continue

            results.append({
                'title': doc_titles[idx],
                'author': doc_authors[idx],
                'date': doc_times[idx],
                'score': score,
                'url': doc_urls[idx],
                'tags': doc_tags[idx]
            })

    else:
        for idx in range(len(doc_titles)):
            author = doc_authors[idx].strip().lower()
            doc_date = str(doc_times[idx])[:10]

            if selected_authors and author not in selected_authors:
                continue
            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue

            results.append({
                'title': doc_titles[idx],
                'author': doc_authors[idx],
                'date': doc_times[idx],
                'score': 0,
                'url': doc_urls[idx],
                'tags': doc_tags[idx]
            })


    total_results = len(results)
    total_pages = ceil(total_results / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]

    # Bangun query string untuk pagination links
    filter_query = f"&query={query}&date_from={date_from}&date_to={date_to}"
    for author in selected_authors:
        filter_query += f"&author_filter={author}"

    return render_template('index.html',
                           results=paginated_results,
                           total_results=total_results,
                           page=page,
                           total_pages=total_pages,
                           selected_authors=selected_authors,
                           authors=all_author,
                           available_years=available_years,
                           date_from=date_from,
                           date_to=date_to,
                           query=query,
                           filter_query=filter_query)



app.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

if __name__ == '__main__':
    app.run(debug=True)