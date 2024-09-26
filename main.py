# Transcripts of developing toddlers (n=40) and their mothers during playtimes.
# Longitudinal data gathered at ages 21, 24, 27, 30, 33, and 36 months.
# Source: https://childes.talkbank.org/access/Eng-NA/Champaign.html

"""
CHI: my finger hurts .
MOT: what's the matter with your finger (.) Adam ?
CHI: ow (.) Mommy (.) ow .
CHI: look what I doing to my finger .
CHI: I mashing it .
URS: but doesn't it hurt (.) Adam ?
CHI: no .
URS: why ?
CHI: I doing very carefully .
URS: don't mash your finger (.) Adam .
CHI: why ?
URS: it will hurt you .
CHI: I mashing my finger .
CHI: my finger hurts .
"""

from collections import Counter
import string


if __name__ == '__main__':
    filename = 'Champaign/21P/01G.cha'

    with open(filename, 'r') as file:
        data = file.read()
        print(data)

    import pylangacq as pla
    import pandas as pd

    corpus = pla.read_chat(filename)

    participants = corpus.participants()
    print(participants)

    utterances = corpus.utterances(participants=['CHI'])  # 'CHI' stands for child
    for utt in utterances:
        print(utt)

    words = corpus.words(participants=['CHI'])
    print(words)

    data = []

    for utterance in corpus.utterances():
        data.append({
            'participant': utterance.participant,
            'utterance': ' '.join(str(token) for token in utterance.tokens),
            'time_marks': utterance.time_marks
        })

    df = pd.DataFrame(data)

    print(df.head())

    word_counts = df.groupby('participant')['utterance'].apply(lambda x: x.str.split().str.len().sum())
    print(word_counts)

    mask = df['participant'] == 'CHI'
    mask &= df['utterance'].str.contains(r'\bdog\b', case=False)
    child_dog_utterances = df[mask]
    print(child_dog_utterances)

    # Step 2: Extract words spoken by the child
    child_words = corpus.words(participants=['CHI'])

    # Step 3: Preprocess the words
    # Convert to lowercase
    child_words = [word.lower() for word in child_words]

    # Remove punctuation and non-word tokens
    punctuation = set(string.punctuation)
    child_words = [word for word in child_words if word not in punctuation and word.isalpha()]

    # Exclude noise tokens
    noise_tokens = {'xxx', 'yyy', 'www'}
    child_words = [word for word in child_words if word not in noise_tokens]

    # Step 4: Count word frequencies
    word_counts = Counter(child_words)

    # Step 5: Get the top 5 most common words
    top_5_words = word_counts.most_common(5)

    print("Top 5 most common words used by the child:")
    for word, count in top_5_words:
        print(f"{word}: {count}")

    # Optional: Exclude stop words
    stop_words = {'the', 'and', 'a', 'is', 'to', 'in', 'it', 'of', 'you', 'that',
                'i', 'for', 'on', 'with', 'was', 'as', 'he', 'be', 'at', 'by',
                'she', 'had', 'but', 'they', 'we', 'his', 'her', 'or', 'an'}

    content_words = [word for word in child_words if word not in stop_words]

    content_word_counts = Counter(content_words)

    top_5_content_words = content_word_counts.most_common(20)

    print("\nTop 20 most common content words used by the child:")
    for word, count in top_5_content_words:
        print(f"{word}: {count}")

    # Now get the top 20 words from the mother's speech

    mother_words = corpus.words(participants=['MOT'])

    mother_words = [word.lower() for word in mother_words]

    mother_words = [word for word in mother_words if word not in punctuation and word.isalpha()]

    mother_words = [word for word in mother_words if word not in noise_tokens]

    mother_content_words = [word for word in mother_words if word not in stop_words]

    mother_content_word_counts = Counter(mother_content_words)

    top_5_mother_content_words = mother_content_word_counts.most_common(20)

    print("\nTop 20 most common content words used by the mother:")

    for word, count in top_5_mother_content_words:
        print(f"{word}: {count}")
