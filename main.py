from pathlib import Path
import pylangacq
import nltk

import analysis
import data_organization
import text_preprocessing
import plotting

def main(studies):
    corpora_paths = [Path(f'data/{study}.zip') for study in studies]
    corpora_names = [path.stem for path in corpora_paths]
    data = {'studies': {s: {'reader': pylangacq.read_chat(str(path))} for s, path in zip(corpora_names, corpora_paths)}}

    data = data_organization.extract_child_ids(data, studies)
    data = data_organization.determine_age_availability(data, studies)
    data = data_organization.organize_by_child_age_and_gender(data, studies)
    analysis.calculate_gender_distribution(data, studies)
    data, ages_of_measurement_buckets = data_organization.extract_transcripts_by_age_and_gender(data, studies)
    plotting.plot_ngrams(data, studies)

    for figure in ['child', 'parent']:
        if figure == 'parent' and studies == ['Garvey']:
            continue  # Garvey corpus does not have parent participants, only children.
        transcripts_by_age = data_organization.bucket_transcripts_by_age_of_measurement(data, ages_of_measurement_buckets, figure, studies)
        processed_texts, nested_processed_texts = text_preprocessing.preprocess_text(transcripts_by_age)
        analysis.do_pca(processed_texts, figure, studies)
        lda, vectorizer, topic_labels = analysis.do_lda(nested_processed_texts, ages_of_measurement_buckets, studies, figure)
        child_transcripts_by_age_of_measurement = data_organization.organize_transcripts_by_age_of_measurement_and_child_id_for_studies(data, studies, figure)
        analysis.do_clustering(child_transcripts_by_age_of_measurement, lda, vectorizer, topic_labels, studies, data, figure)

if __name__ == '__main__':
    nltk.download('stopwords')

    for study_combination in [
        ['Bates'], ['Champaign'], ['Garvey'], ['Hall'], ['HSLLD'],  # Individual studies
        ['Bates', 'Champaign', 'Garvey', 'Hall', 'HSLLD']  # All studies combined
    ]:
        main(study_combination)