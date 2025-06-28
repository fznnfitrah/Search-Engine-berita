# app.py
from flask import Flask, request, render_template
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
import nltk, pandas as pd
import re
from collections import Counter

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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
    df = pd.read_csv("medium_articles.csv", usecols=['title','text','tags','url', 'authors', 'timestamp'], nrows=5000)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    doc_ids = df.index.astype(str).tolist()
    doc_titles = df['title'].tolist()
    doc_urls = df['url'].tolist()
    doc_authors = df['authors'].tolist()
    doc_times = df['timestamp'].dt.strftime('%Y-%m-%d').tolist()

    df['year'] = df['timestamp'].dt.year
    doc_years = df['year'].tolist()
    available_years = sorted(df['year'].dropna().unique().astype(int), reverse=True)


    tokenized_docs = []
    for title, text in zip(df['title'], df['text']):
        words = (str(title) + " " + str(text)).split()
        filtered = [w for w in words if w.lower() not in stop_words]
        tokenized_docs.append(filtered)

    bm25 = BM25Okapi(tokenized_docs)

    tag_set = set()
    cleaned_doc_tags = []
    tag_counter = Counter()

    for tags in df['tags'].fillna(''):
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
def index():
    results = []
    selected_tags = []
    selected_year = '' 


    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        selected_tags = request.form.getlist('tag_filter')
        selected_year = request.form.get('year_filter', '')
        selected_tags = [tag.lower() for tag in selected_tags]

        if query:
            tokens = [w for w in query.split() if w.lower() not in stop_words]
            scores = bm25.get_scores(tokens)
            ranked_indices = [i for i, score in enumerate(scores) if score > 0]
            
            if not ranked_indices:
                ranked_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:100]]

            for idx in ranked_indices:
                tag_text = str(doc_tags[idx]).lower()
                if selected_tags and not any(tag in tag_text for tag in selected_tags):
                    continue
                
                if selected_year and (pd.isna(doc_years[idx]) or int(doc_years[idx]) != int(selected_year)):
                    continue

                results.append({
                    'doc_id': doc_ids[idx],
                    'title': doc_titles[idx],
                    'author': filter_author[idx],
                    'tags': doc_tags[idx],
                    'url': doc_urls[idx],
                    'time': doc_times[idx],
                    'score': f"{scores[idx]:.4f}"
                })
        
        # KASUS 2: PENGGUNA TIDAK MEMASUKKAN KUERI (HANYA FILTER)
        else:
            # Iterasi melalui SEMUA dokumen dari 0 sampai akhir
            for idx in range(len(doc_ids)):
                # Terapkan filter yang dipilih
                tag_text = str(doc_tags[idx]).lower()
                if selected_tags and not any(tag in tag_text for tag in selected_tags):
                    continue
                
                if selected_year and (pd.isna(doc_years[idx]) or int(doc_years[idx]) != int(selected_year)):
                    continue
                
                # Jika lolos filter, tambahkan ke hasil
                results.append({
                    'doc_id': doc_ids[idx],
                    'title': doc_titles[idx],
                    'author': filter_author[idx],
                    'tags': doc_tags[idx],
                    'url': doc_urls[idx],
                    'time': doc_times[idx],
                    'score': 'N/A' # Tidak ada skor karena tidak ada kueri
                })
                
                # Batasi hasil agar tidak terlalu banyak, misal 100 hasil pertama yang cocok
                if len(results) >= 100:
                    break

    # Bagian render_template tetap sama
    return render_template('index.html', results=results, all_tags=all_tags, selected_tags=selected_tags, tag_counts=tag_counts,available_years=available_years,selected_year=selected_year)

app.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

if __name__ == '__main__':
    app.run(debug=True)