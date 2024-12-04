import numpy as np
import pandas as pd
import streamlit as st
from client import name_topic
from documents import documents
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, MiniBatchNMF, NMF

init = "nndsvda"

st.title("Configurable Topic Modeling")
st.write("Customize the topic modeling parameters and explore the results dynamically.")

n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=3, step=1)
n_top_words = st.slider("Number of Top Words per Topic", min_value=5, max_value=30, value=20, step=1)
random_state = st.slider("Random State", min_value=0, max_value=5, value=0, step=1)
algorithm = st.selectbox("Choose Topic Modeling Algorithm", ["LatentDirichletAllocation", "NMF", "MiniBatchNMF"])

if algorithm == "MiniBatchNMF" or algorithm == "NMF":
    beta_loss = st.selectbox("Beta Loss", ["frobenius", "kullback-leibler"])

ages = sorted(documents.keys())
document_texts = [documents[age] for age in ages]

vectorizer = CountVectorizer(stop_words="english")
doc_term_matrix = vectorizer.fit_transform(document_texts)
feature_names = vectorizer.get_feature_names_out()

if algorithm == "LatentDirichletAllocation":
    model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
elif algorithm == "NMF":
    if beta_loss == "frobenius":
        model = NMF(n_components=n_topics, random_state=random_state, init=init, beta_loss=beta_loss, alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1)
    elif beta_loss == "kullback-leibler":
        model = NMF(n_components=n_topics, random_state=random_state, init=init, beta_loss=beta_loss, solver="mu", max_iter=1000, alpha_W=0.00005, alpha_H=0.00005, l1_ratio=0.5)
elif algorithm == "MiniBatchNMF":
    model = MiniBatchNMF(n_components=n_topics, random_state=random_state, init = "nndsvda", batch_size=128, beta_loss=beta_loss, alpha_W=0.00005, alpha_H=0.00005, l1_ratio=0.5)

model.fit(doc_term_matrix)

st.subheader("Discovered Topics")
# on = st.toggle("Remove Common Words", value=True)


# if on:
#     common_words = set()
#     for topic_idx, topic in enumerate(model.components_):
#         for topic_idx2, topic2 in enumerate(model.components_):
#             top_features_indices = topic.argsort()[-n_top_words:][::-1]
#             top_features_indices2 = topic2.argsort()[-n_top_words:][::-1]
#             common_words.update(set([feature_names[i] for i in top_features_indices if feature_names[i] in set([feature_names[j] for j in top_features_indices2])]))

#     topics = {}
#     for topic_idx, topic in enumerate(model.components_):
#         top_features_indices = topic.argsort()[-n_top_words:][::-1]
#         top_features = [feature_names[i] for i in top_features_indices if feature_names[i] not in common_words]
#         topics[f"Topic {topic_idx + 1}"] = ", ".join(top_features)
#         st.write(f"**Topic {topic_idx + 1}**: {', '.join(top_features)}")
# else:
#     topics2 = {}
#     for topic_idx, topic in enumerate(model.components_):
#         top_features_indices = topic.argsort()[-n_top_words:][::-1]
#         top_features = [feature_names[i] for i in top_features_indices]
#         topics2[f"Topic {topic_idx + 1}"] = ", ".join(top_features)
#         st.write(f"**Topic {topic_idx + 1}**: {', '.join(top_features)}")



topics = {}
names = []
for topic_idx, topic in enumerate(model.components_):
    top_features_indices = topic.argsort()[-n_top_words:][::-1]
    top_features = [feature_names[i] for i in top_features_indices]
    topics[f"Topic {topic_idx + 1}"] = ", ".join(top_features)
    names.append(name_topic(top_features).replace('"', ''))
    st.write(f"**{names[topic_idx]}**: {', '.join(top_features)}")



# print(int(ages))
doc_topic_distributions = model.transform(doc_term_matrix)
df_topic_proportions = pd.DataFrame(doc_topic_distributions, index=[int(x) for x in ages], columns=[f"{names[i]}" for i in range(n_topics)])

window_size = st.slider("Smoothing Window Size", min_value=1, max_value=50, value=3, step=1)
df_topic_proportions_smoothed = df_topic_proportions.rolling(window=window_size, center=True).mean()

st.subheader("Smoothed Topic Proportions Across Ages")
st.line_chart(df_topic_proportions_smoothed, x_label='Ages in Months', y_label='Smoothed Topic Proportion')










