from pathlib import Path
import pylangacq
import nltk

import helpers
import plotting

def main(studies):
    corpora_paths = [Path(f'data/{study}.zip') for study in studies]
    corpora_names = [path.stem for path in corpora_paths]
    data = {'studies': {s: {'reader': pylangacq.read_chat(str(path))} for s, path in zip(corpora_names, corpora_paths)}}

    data = helpers.extract_child_ids(data, studies)
    data = helpers.determine_age_availability(data, studies)
    data = helpers.organize_by_child_age_and_gender(data, studies)
    helpers.calculate_gender_distribution(data, studies)
    data, ages_of_measurement_buckets = helpers.extract_transcripts_by_age_and_gender(data, studies)
    plotting.plot_ngrams(data, studies)

    for figure in ['child', 'parent']:
        if figure == 'parent' and studies == ['Garvey']:
            continue
        transcripts_by_age = helpers.bucket_transcripts_by_age_of_measurement(data, ages_of_measurement_buckets, figure, studies)
        processed_texts, processed_texts_NOT_FLAT = helpers.preprocess(transcripts_by_age)
        helpers.do_pca(processed_texts, figure, studies)
        lda, vectorizer, topic_labels = helpers.do_lda(processed_texts_NOT_FLAT, ages_of_measurement_buckets, studies, figure)
        child_transcripts_by_age_of_measurement = helpers.organize_transcripts_by_age_of_measurement_and_child_id_for_studies(data, studies, figure)
        helpers.do_clustering(child_transcripts_by_age_of_measurement, lda, vectorizer, topic_labels, studies, data, figure)

if __name__ == '__main__':
    nltk.download('stopwords')

    for study_combination in [
        ['Bates'], ['Champaign'], ['Garvey'], ['Hall'], ['HSLLD'],
        ['Bates', 'Champaign', 'Garvey', 'Hall', 'HSLLD']
    ]:
        main(study_combination)
