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

# Prof. feedback:
# - You may need to manually interpret the clusters found - which you can receive credit for that.
# - Additionally, analyzing how the cluster membership change over time for different children can potentially answer question (b).
# - Try to provide more details into the experiments in your final report and presentation.
# - Try to think about what sort of evaluation metric will be needed to evaluate whether your project is successful or not.
# - [ ] Possible metric as to whether our project is successful: how well does it align with language acquisition literatu
#
# Prof. advice to entire class:
# - Things to report:
#     - How much did we learn from this?
#     - What are the research questions? Be hypothesis-driven.
#     - **Focus on experinemtal results: how well can we conduct a project like this? Why does it work or not work?
#     - Always have a research question in mind when writing experimental results. Discuss whether you were able to answer the question or not, and explain w
#
# My notes:
# - [x] Decided to not use any part-of-speech tagging.
# - [ ] Identify sets of topics and concepts children (together and/or by gender) and parents (separately) speak about.
#       - [x] Used an LLM help us identify topics/concepts among a set of words.
#       - [ ] Compare n-gram models between that of the child and that of the parent.
# - [x] Explore a small subset of the data to determine if there are a meaningful amount of instances wherein there is a shift in a child's perception of reality. It not, proceed to the next step.
# - [ ] Examine whether, in child-to-child interactions, children use certain parts-of-speech more or less than if they were talking to a parent.
#       - [ ] Account for the difference in setting before drawing conclusions about what this says about child-to-child vs. child-to-parent interactio
# - [ ] BE CAREFUL: is it safe to attribute clusters to children's nature when the setting could be the actual cluster(s)?