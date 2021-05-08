from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as cluster_hierarchy
import scipy.spatial
import streamlit as st
from matplotlib import pyplot as plt


# streamlit run streamlit_app.py


def main():
    df_topics, df_frequencies, df_article_topic, figure_topics_structure = get_data()

    tabs = ['Topics', 'Similarity search', 'Article points', 'Topics structure']
    option_tab = st.sidebar.radio('Navigation', tabs)

    st.sidebar.markdown('_Data sources: tut.by, naviny.online_')

    st.title('Topic modeling :mag: :newspaper: :heavy_check_mark:')

    topic_id = get_topic_id(df_topics) if option_tab in tabs[:3] else None

    if option_tab == tabs[0]:

        st.subheader('Topic popularity over time:')

        if topic_id:
            write_plot(df_frequencies, topic_id)
            write_articles(df_article_topic, topic_id)
        else:
            write_plot_entire(df_topics, df_frequencies)

    elif option_tab == tabs[1]:
        similarity_search(df_article_topic, topic_id)

    elif option_tab == tabs[2]:
        write_plot_points(df_article_topic, topic_id)

    elif option_tab == tabs[3]:
        st.subheader('Topic tree:')
        st.pyplot(figure_topics_structure)


@st.cache(allow_output_mutation=True, persist=False, show_spinner=False)
def get_data():

    path_base = Path(r'data')

    path_topics = path_base / 'topics.parquet'
    path_frequencies = path_base / 'frequencies.parquet'
    path_article_topic = path_base / 'article_topic.parquet'
    path_points = path_base / 'points.parquet'

    df_topics = pd.read_parquet(path_topics)
    df_frequencies = pd.read_parquet(path_frequencies)
    df_article_topic = pd.read_parquet(path_article_topic)
    df_points = pd.read_parquet(path_points)

    df_article_topic = df_article_topic \
        .merge(df_points, on='url_id') \
        .merge(df_topics, on='topic_id') \
        .sort_values(by='time', ascending=False)

    data_topics_structure = get_data_topics_structure(df_article_topic, df_topics)

    return df_topics, df_frequencies, df_article_topic, data_topics_structure


def get_topic_id(df_topics):

    n_articles = df_topics['topic_size'].sum()
    df_topics = df_topics.to_numpy()
    df_topics = np.insert(df_topics, 0, values=[None, n_articles, ['All']], axis=0)

    format_func = lambda record: ', '.join(record[2]) + f' ({record[1]})'
    option = st.selectbox('Topic', df_topics, format_func=format_func)

    topic_id = option[0]

    return topic_id


def write_plot_entire(df_topics, df_frequencies):

    topic_ids = df_topics['topic_id']

    figure = go.Figure()

    for topic_id in topic_ids:
        df_topic_frequencies = df_frequencies[df_frequencies['topic_id'] == topic_id]

        max_change = df_topic_frequencies['frequency'].max() - df_topic_frequencies['frequency'].min()

        if max_change > 0.06:
            topic_words = df_topics[df_topics['topic_id'] == topic_id]['topic_words'].iloc[0]
            topic_words = topic_words[:3]
            topic_words = ', '.join(topic_words)
            figure.add_scatter(x=df_topic_frequencies['time'], y=df_topic_frequencies['frequency'],
                               mode='lines', name=topic_words)

    figure.update_yaxes(title_text='fraction of all articles')
    st.plotly_chart(figure, use_container_width=True)


def write_plot(df_frequencies, topic_id):

    df_topic_frequencies = df_frequencies[df_frequencies['topic_id'] == topic_id]

    figure = go.Figure()
    figure.add_scatter(x=df_topic_frequencies['time'], y=df_topic_frequencies['frequency'],
                       mode='lines')
    figure.update_yaxes(title_text='fraction of all articles')
    st.plotly_chart(figure, use_container_width=True)


def write_articles(df_article_topic, topic_id):

    df_topic = df_article_topic[df_article_topic['topic_id'] == topic_id]
    n_articles = 17
    df_topic = df_topic.head(n_articles)

    st.subheader('Latest articles:')

    for record in df_topic.itertuples():
        tags = ', '.join(record.tags)
        tags = f'({tags})' if tags else ''
        s = fr"_\[{record.time}\]_ **{record.header}** {tags}"
        st.info(s)


def write_plot_points(df_article_topic, topic_id):

    df_topic = df_article_topic[df_article_topic['topic_id'] == topic_id]
    df_not_topic = df_article_topic[df_article_topic['topic_id'] != topic_id]
    df = df_not_topic.sample(5000).append(df_topic)

    points = df['point'].to_list()
    points = np.array(points)

    hover_data = {
        'header': df['header'],
        'tags': df['tags'],
        'topic': df['topic_words'],
    }

    color = (df['topic_id'] == topic_id) if topic_id else df['topic_id']

    figure = px.scatter(
        x=points[:, 0],
        y=points[:, 1],
        color=color,
        color_continuous_scale=px.colors.cyclical.IceFire,
        hover_data=hover_data,
    )

    figure.update_traces(marker=dict(size=4), showlegend=False)

    st.subheader('Each point is an article in a semantic space.')
    st.plotly_chart(figure, use_container_width=True)


def similarity_search(df_article_topic, topic_id):

    if topic_id:
        df_article_topic = df_article_topic[df_article_topic['topic_id'] == topic_id]

    options = df_article_topic.iloc[:1000][['url_id', 'header']].to_numpy()
    format_func = lambda record: record[1]
    option = st.selectbox('Article', options, format_func=format_func)
    url_id = option[0]

    record = df_article_topic[df_article_topic['url_id'] == url_id].iloc[0]
    embedding = record['embedding_document']
    embedding = embedding[np.newaxis, ...]
    embedding = embedding.astype(np.float32)

    embeddings = df_article_topic['embedding_document'].to_numpy()
    embeddings = np.stack(embeddings)

    distances = scipy.spatial.distance.cdist(embedding, embeddings, metric='cosine')[0]
    n_articles = 5
    index = distances.argsort()[:n_articles]

    records = df_article_topic.iloc[index]
    distances = distances[index]

    st.subheader('The most similar articles to the selected one:')

    for record, distance in zip(records.iterrows(), distances):
        record = record[1]
        s = fr"_\[{record['time']}\]_ **{record['header']}** [{', '.join(record['tags'])}] \[{', '.join(record['topic_words'])}\] (distance: {distance:.3f})"
        st.info(s)


def get_data_topics_structure(df_acticles, df_topics):

    f = lambda group: np.mean(group.to_list(), axis=0, dtype=float).tobytes()

    embeddings = df_acticles.groupby('topic_id')['embedding_document'].agg(f).map(np.frombuffer)
    topic_ids = embeddings.index.to_numpy()
    embeddings = embeddings.to_list()
    embeddings = np.array(embeddings)

    topic_words = df_topics.set_index('topic_id')['topic_words'][topic_ids]
    topic_words = topic_words.map(lambda words: ', '.join(words)).values

    linkage = cluster_hierarchy.linkage(embeddings, method='complete', metric='cosine',
                                        optimal_ordering=True)

    figure, ax = plt.subplots(figsize=(3, 30))

    dendrogram = cluster_hierarchy.dendrogram(linkage, labels=topic_words, orientation='left',
                                              leaf_font_size=12, no_plot=False, ax=ax)

    return figure


if __name__ == '__main__':
    main()
