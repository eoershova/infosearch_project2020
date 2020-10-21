from flask import Flask, request, render_template
from search_engine import *

app = Flask(__name__)

get_model()
get_mystem()
get_queries()
get_answers()
get_stop_words()
query_base_indexing_bm25()
query_base_indexing_tfidf()
query_base_indexing_w2v_basic()
query_base_indexing_w2v_advanced()


def search(query, search_type):

    preproc_query = preprocess(query)
    if preproc_query == '':
        answer = 'Сформулируйте вопрос иначе, пожалуйста. Опишите ситуацию подробнее.'
        return [answer]

    if search_type == 0:
        answer = 'Выберите метрику для поиска'
    elif search_type == 1:
        answer = bm25_search(query)
    elif search_type == 2:
        answer = tfidf_search(query)
    elif search_type == 3:
        answer = w2v_basic_search(query)
    elif search_type == 4:
        answer = w2v_advanced_search(query)
    return [answer]


@app.route('/')
def index():
    if request.args:
        query = request.args['search']
        search_type = int(request.args['search_type'])
        answer = search(query, search_type)
        return render_template('index.html', answer=answer)
    return render_template('index.html', links=[])


if __name__ == '__main__':
    app.run(debug=True)
