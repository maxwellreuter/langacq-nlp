import re
import nltk

def preprocess_documents(document_texts, flatten=True):
    """
    Preprocess documents by removing stopwords, punctuation, and common words.

    Args:
        document_texts (list): A nested list of input document texts.
        flatten (bool): Whether to flatten the nested list of document texts.

    Returns:
        list: A list of preprocessed document texts.
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


def preprocess_text(child_transcripts_by_age):
    """
    Preprocess transcripts by age, returning both flat and nested formats.

    Args:
        child_transcripts_by_age (dict): A dictionary organized by age of measurement and child ID.

    Returns:
        list: A list of preprocessed document texts.
        list: A nested list of preprocessed document texts.
    """
    texts_by_age = [child_transcripts_by_age[age] for age in sorted(child_transcripts_by_age)]
    return preprocess_documents(texts_by_age, flatten=True), preprocess_documents(texts_by_age, flatten=False)