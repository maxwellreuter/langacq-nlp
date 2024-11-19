from collections import Counter
import pylangacq
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
import nltk
import os
import math
import numpy as np
from openai import OpenAI

import plotting

def extract_participant_ids(study, reader):
    # Extract the filenames without path or extension
    filenames = [re.search(r'(?<=\/)[^\/]+(?=\.cha)', path).group() for path in reader.file_paths()]
    
    if study == 'Bates':        
        cleaned_filenames = []
        for filename in filenames:
            filename = re.sub(r'\d+',       '', filename)  # Remove numbers
            filename = re.sub(r'st$',       '', filename)  # Remove 'st' if it appears at the end
            filename = re.sub(r'snack$',    '', filename)  # Remove 'snack' if it appears at the end
            filename = re.sub(r'[^a-zA-Z]', '', filename)  # Remove non-alphabetical characters
            cleaned_filenames.append(filename)
        
        # Filter filenames that don't appear in all 4 sessions
        return sorted([filename for filename, count in Counter(cleaned_filenames).items() if count == 4])
    elif study == 'Champaign':
        # Filter filenames that don't appear in all 12 sessions
        return sorted([filename for filename, count in Counter(filenames).items() if count == 12])
    elif study == 'HSLLD':
        return sorted(list(set([filename[:3] for filename in filenames])))
    return sorted(list(set(filenames)))

def extract_child_ids(data, study):
    # Create the output directory if it doesn't exist
    output_dir = f"results/{study}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the file for writing
    output_file = f"{output_dir}/children_in_full_longitude.txt"
    with open(output_file, "w") as f:
        f.write('Number of children present for the full longitude of the study:\n')
        total = 0
        for study in data['studies'].keys():
            data['studies'][study]['children'] = {}
            child_ids = extract_participant_ids(study, data['studies'][study]['reader'])
            f.write(f'    {study:<9} n={len(child_ids)}\n')
            for child_id in child_ids:
                data['studies'][study]['children'][child_id] = {'transcripts': {}}
                total += 1
        f.write(f'    ---------------\n    {"Total":<9} N={total}\n')

    return data

def determine_age_availability(data, study):
    # Determine the ages at which data is available, rounding to the nearest month.

    # Calculate unique rounded ages.
    ages_of_measurement = sorted(
        round(age_of_measurement) 
        for data in data['studies'].values() 
        if 'reader' in data 
        for age_of_measurement in data['reader'].ages(months=True) 
        if age_of_measurement not in [None, 0]
    )
    data['ages_of_measurement'] = {age_of_measurement: {
        'male':   {'readers': [], 'transcripts': []},
        'female': {'readers': [], 'transcripts': []},
        } for age_of_measurement in ages_of_measurement}

    # Plot the number of children in each age bucket.
    age_of_measurement_counter = Counter(ages_of_measurement)
    ages = [str(age) for age, _ in age_of_measurement_counter.items()]
    counts = [count for _, count in age_of_measurement_counter.items()]

    plotting.plot_age_availability(ages, counts, study)

    return data

def organize_by_child_age_and_gender(data, study):
    num_invalid_ages, num_valid_ages, num_invalid_genders, num_valid_genders = 0, 0, 0, 0

    output_dir = f"results/{study}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/discard_summary.txt"

    data['genders'] = {'male': pylangacq.Reader(), 'female': pylangacq.Reader()}

    for study_data in data['studies'].values():
        for child_id, child_data in study_data['children'].items():
            child_reader = pylangacq.read_chat(f'data/{study}.zip', match=child_id)
            gender = (child_reader.headers()[0].get('Participants', {}).get('CHI', {}).get('sex') or None)
            child_data['gender'] = gender
            num_valid_genders += bool(gender)
            num_invalid_genders += not bool(gender)
            if not gender: continue

            for age_reader in child_reader:
                ages = age_reader.ages(months=True)
                if len(ages) != 1 or ages[0] in [None, 0]:
                    num_invalid_ages += 1
                    continue
                age = round(ages[0])
                num_valid_ages += 1
                data['ages_of_measurement'][age][gender]['readers'].append(age_reader)
                data['genders'][gender].append(age_reader)
                child_data['transcripts'].setdefault(age, []).append(age_reader)

    with open(output_file, "w") as f:
        if num_valid_ages + num_invalid_ages:
            f.write(f'Discarded {num_invalid_ages} readers with invalid ages ({round(100 * num_invalid_ages / (num_valid_ages + num_invalid_ages))}%).\n')
        if num_valid_genders + num_invalid_genders:
            f.write(f'Discarded {num_invalid_genders} invalid genders ({round(100 * num_invalid_genders / (num_valid_genders + num_invalid_genders))}%).\n')

    return data

def calculate_gender_distribution(data, study):
    output_dir = f"results/{study}"
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

def extract_transcripts_by_age_and_gender(data, study):
    output_dir = f"results/{study}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/transcript_counts.txt"

    male_transcripts_count = female_transcripts_count = num_child_words = num_parent_words = 0

    for age, age_data in data['ages_of_measurement'].items():
        for gender in ['male', 'female']:
            transcripts = [
                {
                    'child': ' '.join(reader.words(participants='CHI')),
                    'parent': ' '.join(reader.words(participants=['MOT', 'FAT']))
                }
                for reader in age_data[gender]['readers']
            ]
            age_data[gender]['transcripts'] = transcripts
            num_child_words += sum(len(t['child'].split()) for t in transcripts)
            num_parent_words += sum(len(t['parent'].split()) for t in transcripts)
            if gender == 'male':
                male_transcripts_count += len(transcripts)
            else:
                female_transcripts_count += len(transcripts)

    total_transcripts = male_transcripts_count + female_transcripts_count
    percent_male = round(100 * male_transcripts_count / total_transcripts)
    percent_female = 100 - percent_male
    total_words = num_child_words + num_parent_words
    percent_child_words = round(100 * num_child_words / total_words)

    with open(output_file, "w") as f:
        f.write('\nTotal number of transcripts:\n')
        f.write(f'    Male   child: {male_transcripts_count} ({percent_male}%)\n')
        f.write(f'    Female child: {female_transcripts_count} ({percent_female}%)\n')
        f.write('\nTotal number of words:\n')
        f.write(f'    Child words: {num_child_words}\n')
        f.write(f'    Parent words: {num_parent_words}\n')
        f.write(f'    Percent child words: {percent_child_words}%\n')

    return data, data['ages_of_measurement'].keys()

def bucket_transcripts_by_age_of_measurement(data, ages_of_measurement_buckets, participants, study):
    # Create the output directory if it doesn't exist
    output_dir = f"results/{study}"
    os.makedirs(output_dir, exist_ok=True)

    # File to save the bucket summary
    output_file = f"{output_dir}/transcript_buckets_{participants}.txt"

    # Bucket child transcripts by age of measurement
    child_transcripts_by_age = {
        age_of_measurement: [
            item[participants] for gender in ['male', 'female']
            for item in data['ages_of_measurement'][age_of_measurement][gender]['transcripts']
        ]
        for age_of_measurement in ages_of_measurement_buckets
        if data['ages_of_measurement'][age_of_measurement]
    }

    num_total_transcripts = sum(len(child_transcripts) for child_transcripts in child_transcripts_by_age.values())
    num_buckets = len(child_transcripts_by_age)
    average_transcripts_per_bucket = round(num_total_transcripts / num_buckets)

    # Write the summary to the file
    with open(output_file, "w") as f:
        f.write(f'Organized {num_total_transcripts} transcripts into {num_buckets} buckets (~{average_transcripts_per_bucket}/bucket) ({participants}).\n')

    return child_transcripts_by_age

def preprocess_documents(document_texts, flatten: bool = True):
    """
    Preprocess documents by removing stopwords, punctuation, and common words.
    """
    omit = set(nltk.corpus.stopwords.words('english')).union({
        # Yes/no variants
        'yes', 'yep', 'yea', 'yeah', 'uhhuh', 'mhm', 'okay', 'no', 'nope', 'na',
        # Exclamations
        'oh', 'ah', 'mm', 'hm', 'huh', 'uhoh',
        # Hi/bye
        'hi', 'hey', 'bye',
        # Parent references
        'mom', 'mama', 'mommy', 'dad', 'dada', 'daddy',
        # Common words
        'thats', 'get', 'got', 'like', 'want', 'put', 'see', 'go', 'right', 'look', 'hafta', 'wanna', 
        # Common verbs
        'dont', 'gonna',
        # Numbers
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        # Noises
        'zzz',
        # Unfamiliar tokens
        '0is'
    })

    preprocess_text = lambda text: " ".join(
        word for word in re.sub(r'[^\w\s]', '', text.lower()).split() if word not in omit
    ) if isinstance(text, str) else ""

    if flatten:
        document_texts = [doc for sublist in document_texts for doc in sublist] if isinstance(document_texts[0], list) else document_texts
    return [
        [preprocess_text(doc) for doc in sublist] if isinstance(sublist, list) else preprocess_text(sublist)
        for sublist in document_texts
    ] if not flatten else [preprocess_text(doc) for doc in document_texts]

def preprocess(child_transcripts_by_age):
    """
    Preprocess transcripts by age, returning both flat and nested formats.
    """
    texts_by_age = [child_transcripts_by_age[age] for age in sorted(child_transcripts_by_age)]
    return preprocess_documents(texts_by_age, flatten=True), preprocess_documents(texts_by_age, flatten=False)

def do_pca(processed_texts, participant, study):
    # Perform PCA to help determine a good number of components (i.e. topics/concepts) to use for LDA.
    def perform_pca(doc_term_matrix, participant):
        # Perform PCA
        pca = PCA()
        pca.fit(doc_term_matrix.toarray())  # Convert sparse matrix to dense for PCA
        
        # Plot the eigenvalues (explained variance) for the first 90 components
        explained_variance = pca.explained_variance_ratio_

        plotting.plot_pca(explained_variance, participant, study)

    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    perform_pca(doc_term_matrix, participant)

# Predict Topics for New Transcripts
def predict_topic(lda, vectorizer, new_transcripts):
    # Transform new transcripts into the LDA's vectorizer space
    X_new = vectorizer.transform(new_transcripts)

    # Get topic probabilities for each new transcript
    topic_distributions = lda.transform(X_new)

    # Assign the topic with the highest probability
    predicted_topics = np.argmax(topic_distributions, axis=1)

    return predicted_topics, topic_distributions

def do_lda(document_texts, ages, study, participant):
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
        with open(f"results/{study}/topics.txt", "w") as f:
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

        # Compute document-topic distributions
        doc_topic_distributions = lda.transform(doc_term_matrix)

        # Flatten document texts for age mapping
        expanded_ages = [age for age, sublist in zip(ages, nested_texts) for _ in sublist]

        assert len(expanded_ages) == doc_topic_distributions.shape[0], \
            f"Length mismatch: expanded_ages ({len(expanded_ages)}) and doc_topic_distributions ({doc_topic_distributions.shape[0]})"

        # Create DataFrame
        df = pd.DataFrame(doc_topic_distributions, index=expanded_ages, columns=topic_labels)
        df_topic_proportions = df.groupby(level=0).mean()

        # Plot smoothed topic proportions
        smoothed = df_topic_proportions.rolling(window=window_size, center=True).mean()
        plotting.plot_lda(smoothed, participant, study, window_size)

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

def organize_transcripts_by_age_of_measurement_and_child_id_for_study(data, study, participant):
    # Organize transcripts by age of measurement and child ID for a specific study.

    study_ages_of_measurement = set()
    for study in [study]:
        for child_id in list(data['studies'][study]['children'].keys()):
            study_ages_of_measurement.update(data['studies'][study]['children'][child_id]['transcripts'].keys())

    child_transcripts_by_age_of_measurement = {age: {} for age in study_ages_of_measurement}
    for study in [study]:
        for age in study_ages_of_measurement:
            for child in list(data['studies'][study]['children'].keys()):
                if age in data['studies'][study]['children'][child]['transcripts']:
                    child_transcripts_by_age_of_measurement[age][child] = preprocess_documents([' '.join(reader.words(participants='CHI' if participant == 'child' else ['MOT', 'FAT'])) for reader in data['studies'][study]['children'][child]['transcripts'][age]], flatten=True)
                else:
                    child_transcripts_by_age_of_measurement[age][child] = []

    return child_transcripts_by_age_of_measurement

def do_clustering(child_transcripts_by_age_of_measurement, lda, vectorizer, topic_labels, study, data, participant):
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
            lda, vectorizer, age_transcripts, most_recent_locations, age, topic_labels, study, data, participant
        )
