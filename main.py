from pathlib import Path
import pylangacq
import nltk

import helpers
import plotting

def main(STUDY):
    if STUDY == 'All':
        corpora_paths = list(Path('data').glob('*.zip'))
    else:
        corpora_paths = [Path(f'data/{STUDY}.zip')]
    corpora_names = [path.stem for path in corpora_paths]
    data = {'studies': {s: {'reader': pylangacq.read_chat(str(path))} for s, path in zip(corpora_names, corpora_paths)}}

    data = helpers.extract_child_ids(data, STUDY)
    data = helpers.determine_age_availability(data, STUDY)
    data = helpers.organize_by_child_age_and_gender(data, STUDY)
    helpers.calculate_gender_distribution(data, STUDY)
    data, ages_of_measurement_buckets = helpers.extract_transcripts_by_age_and_gender(data, STUDY)
    plotting.plot_ngrams(data, STUDY)

    for figure in ['child', 'parent']:
        if study in ['All', 'Garvey'] and figure == 'parent':
            print('Skipping parent figures for Garvey/All.')
            continue

        transcripts_by_age = helpers.bucket_transcripts_by_age_of_measurement(data, ages_of_measurement_buckets, figure, STUDY)
        processed_texts, processed_texts_NOT_FLAT = helpers.preprocess(transcripts_by_age)
        helpers.do_pca(processed_texts, figure, STUDY)
        lda, vectorizer, topic_labels = helpers.do_lda(processed_texts_NOT_FLAT, ages_of_measurement_buckets, STUDY, figure)
        child_transcripts_by_age_of_measurement = helpers.organize_transcripts_by_age_of_measurement_and_child_id_for_study(data, STUDY, figure)
        helpers.do_clustering(child_transcripts_by_age_of_measurement, lda, vectorizer, topic_labels, STUDY, data, figure)

if __name__ == '__main__':
    nltk.download('stopwords')

    for study in ['Bates', 'Champaign', 'Garvey', 'Hall', 'HSLLD']:
        main(study)
