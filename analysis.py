from collections import Counter
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
import os
import math
import numpy as np
from openai import OpenAI

import plotting

def calculate_gender_distribution(data, studies):
    """
    Calculate the gender distrubution for the given studies.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        studies (list of strings): The names of the studies (e.g. Bates, Champaign, etc.).

    Returns:
        None (outputs are written to a file).
    """
    # Create a file for writing the gender distrubution for the given studies
    output_dir = f"results/{'_'.join(studies)}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/gender_distribution.txt"

    num_total_males = num_total_females = num_total_unspecified = 0
    with open(output_file, "w") as f:
        f.write('Gender distribution per study:\n')

        for name, corpus in data['studies'].items():
            gender_counts = Counter(
                child['gender'] if child['gender'] in ['male', 'female'] else 'unspecified'
                for child in corpus['children'].values()
            )
            num_males, num_females, num_unspecified = (
                gender_counts.get('male', 0),
                gender_counts.get('female', 0),
                gender_counts.get('unspecified', 0),
            )
            total = num_males + num_females + num_unspecified
            percentages = [round(100 * count / total) for count in (num_males, num_females, num_unspecified)]
            num_total_males += num_males
            num_total_females += num_females
            num_total_unspecified += num_unspecified

            f.write(f'    {name:<9} {percentages[0]}%m, {percentages[1]}%f ({percentages[2]}% unsp.)\n')

        total_all = num_total_males + num_total_females + num_total_unspecified
        total_percentages = [round(100 * count / total_all) for count in (num_total_males, num_total_females, num_total_unspecified)]

        f.write('    ---------------------------------\n')
        f.write(f'    {"Total":<9} {total_percentages[0]}%m, {total_percentages[1]}%f ({total_percentages[2]}% unsp.)\n')


def do_pca(processed_texts, participant, studies):
    """
    Perform PCA to help determine a good number of components (i.e. topics/concepts) to use for LDA.

    Args:
        processed_texts (list): A list of preprocessed document texts.
        participant (str): The participant type ('child' or 'parent').
        studies (list): A list of studies.

    Returns:
        None (plots the PCA results).
    """
    def perform_pca(doc_term_matrix, participant):
        pca = PCA()
        pca.fit(doc_term_matrix.toarray())  # Convert sparse matrix to dense for PCA
        
        # Plot the eigenvalues (explained variance) for the first 90 components
        explained_variance = pca.explained_variance_ratio_

        plotting.plot_pca(explained_variance, participant, studies)

    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    perform_pca(doc_term_matrix, participant)


def predict_topic(lda, vectorizer, new_transcripts):
    """
    Predict topics for new transcripts.

    Args:
        lda (LatentDirichletAllocation): The LDA model.
        vectorizer (CountVectorizer): The vectorizer used to transform the data.
        new_transcripts (list): A list of new transcripts to predict topics for.

    Returns:
        predicted_topics (numpy.ndarray): The predicted topics for the new transcripts.
        topic_distributions (numpy.ndarray): The topic distributions for the new transcripts.
    """
    # Transform new transcripts into the LDA's vectorizer space
    X_new = vectorizer.transform(new_transcripts)

    # Get topic probabilities for each new transcript
    topic_distributions = lda.transform(X_new)

    # Assign the topic with the highest probability
    predicted_topics = np.argmax(topic_distributions, axis=1)

    return predicted_topics, topic_distributions


def do_lda(document_texts, ages, studies, participant):
    """
    Perform LDA on the given document_texts and display results.

    Args:
        document_texts (list): Nested list of document texts (not flat).
        ages (list): List of ages corresponding to document_texts.
        study (str): Study name.
        participant (str): Participant name.

    Returns:
        tuple: LDA model, vectorizer, and topic labels.
    """
    def flatten_nested_texts(nested_texts):
        """Flatten a nested list of texts."""
        return [doc for sublist in nested_texts for doc in sublist]

    def train_lda(flat_texts, n_topics):
        """Train LDA model on flat texts."""
        vectorizer = CountVectorizer()
        doc_term_matrix = vectorizer.fit_transform(flat_texts)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(doc_term_matrix)
        return lda, vectorizer, doc_term_matrix

    def display_results(lda, vectorizer, doc_term_matrix, ages, nested_texts, window_size, participant, n_top_words=20):
        """Display LDA results and assign topic labels."""
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        # Open a text file to write the topics to
        with open(f"results/{'_'.join(studies)}/topics_{participant}.txt", "w") as f:
            for topic_idx, topic in enumerate(lda.components_):
                # Extract the top features for each topic
                top_features = [feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]]
                label = f'Topic {topic_idx}'
                topics.append((label, top_features))
                f.write(f"{label}: {', '.join(top_features)}\n")

        topic_words = ""
        topic_words += "\nTopic 1: " + ', '.join(topics[0][1])
        topic_words += "\nTopic 2: " + ', '.join(topics[1][1])
        topic_words += "\nTopic 3: " + ', '.join(topics[2][1])
        topic_words += "\nTopic 4: " + ', '.join(topics[3][1])

        try:
            client = OpenAI(api_key="key_goes_here")

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"The following are features gathered during clustering when analyzing a study concerning child language acquisition. Based on the words in the clusters, assign 1 theme (no more than 2 words) to each of the {len(topic_words)} topics: {topics}. Put the themes inside double carets (<< >>)."
                    }
                ]
            )

            pattern = r"<<(.*?)>>"
            topic_labels = re.findall(pattern, completion.choices[0].message.content)
        except:
            topic_labels = ['NO_OPENAI_KEY_PROVIDED_1', 'NO_OPENAI_KEY_PROVIDED_2', 'NO_OPENAI_KEY_PROVIDED_3', 'NO_OPENAI_KEY_PROVIDED_4']

        # Compute document-topic distributions
        doc_topic_distributions = lda.transform(doc_term_matrix)

        # Flatten document texts for age mapping
        expanded_ages = [age for age, sublist in zip(ages, nested_texts) for _ in sublist]

        assert len(expanded_ages) == doc_topic_distributions.shape[0], \
            f"Length mismatch: expanded_ages ({len(expanded_ages)}) and doc_topic_distributions ({doc_topic_distributions.shape[0]})"

        # Create DataFrame
        df = pd.DataFrame(doc_topic_distributions, index=expanded_ages, columns=topic_labels)
        df_topic_proportions = df.groupby(level=0).mean()

        # Extend data for rolling
        padding = window_size // 2
        first_row = df_topic_proportions.iloc[0]
        last_row = df_topic_proportions.iloc[-1]
        pad_start = pd.DataFrame([first_row] * padding, index=[-i for i in range(padding, 0, -1)])
        pad_end = pd.DataFrame([last_row] * padding, index=[i + 1 for i in range(df_topic_proportions.index[-1], df_topic_proportions.index[-1] + padding)])

        # Combine padded and original data
        extended_df = pd.concat([pad_start, df_topic_proportions, pad_end])

        # Plot smoothed topic proportions
        smoothed = extended_df.rolling(window=window_size, center=True).mean()
        smoothed_trimmed = smoothed.iloc[padding:-padding]  # Optional trimming if only the original range is needed
        plotting.plot_lda(smoothed_trimmed, participant, studies, window_size)

        return topic_labels

    # Flatten document_texts for LDA processing
    flat_texts = flatten_nested_texts(document_texts)

    # Train LDA model
    lda, vectorizer, doc_term_matrix = train_lda(flat_texts, n_topics=4)

    # Display LDA results
    topic_labels = display_results(
        lda, vectorizer, doc_term_matrix, ages, document_texts,
        window_size=math.floor(len(set(ages)) / 3),
        participant=participant
    )

    return lda, vectorizer, topic_labels


def do_clustering(child_transcripts_by_age_of_measurement, lda, vectorizer, topic_labels, studies, data, participant):
    """
    Perform clustering on the given transcripts and display results.

    Args:
        child_transcripts_by_age_of_measurement (dict): Transcripts organized by age of measurement and child ID.
        lda (LatentDirichletAllocation): The LDA model.
        vectorizer (CountVectorizer): The vectorizer used to transform the data.
        topic_labels (list): A list of topic labels.
        studies (list): A list of studies.
        data (dict): The dataset containing study, child, and transcript information.
        participant (str): The participant type ('child' or 'parent').

    Returns:
        None (plots the clustering results).
    """
    # Create topic labels from the first two words of each topic
    most_recent_locations = {}
    for age in sorted(child_transcripts_by_age_of_measurement.keys()):
        age_transcripts = {}
        for child_id in sorted(child_transcripts_by_age_of_measurement[age].keys()):
            # Flatten transcripts for each child
            flattened_transcripts = ' '.join(child_transcripts_by_age_of_measurement[age][child_id])
            if flattened_transcripts.strip():  # Only add if there's actual text
                age_transcripts[child_id] = flattened_transcripts

        # Call the updated function with topic labels
        plotting.plot_participant_topics_on_simplex_with_tracking(
            lda, vectorizer, age_transcripts, most_recent_locations, age, topic_labels, studies, data, participant
        )