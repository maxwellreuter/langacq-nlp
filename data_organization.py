from collections import Counter
import pylangacq
import re
import os

import text_preprocessing
import plotting

def extract_child_ids(data, studies):
    """
    Extract the names/IDs of children in the studies that are present for all sessions of their respective study.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        studies (list of strings): The names of the studies (e.g. Bates, Champaign, etc.).

    Returns:
        data (dict): The input dataset updated with child IDs.
    """
    def extract_child_ids_for_study(study, reader):
        """
        Extract the names/IDs of children in the study that are present for all sessions (i.e. the full longitude).
        Since most studies have different directory structures, each must be handled differently.

        Args:
            study (string): The name of the study (e.g. Bates, Champaign, etc.).
            reader (pylangacq.Reader): The reader containing data on the study.

        Returns:
            list of strings: a sorted list of the names/IDs of children in the study.
        """
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
            # Luckily, the IDs here are the first 3 characers in the filename
            return sorted(list(set([filename[:3] for filename in filenames])))
        
        # For the other studies, returning the unique filenames is sufficient
        return sorted(list(set(filenames)))

    # Create a file for writing the number of children present for the full longitude of the study
    output_dir = f"results/{'_'.join(studies)}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/children_in_full_longitude.txt"
    
    # For each study, extract the child IDs and write the # of children present for the full longitude of the study
    with open(output_file, "w") as f:
        f.write('Number of children present for the full longitude of the study:\n')
        total = 0
        for study in data['studies'].keys():
            data['studies'][study]['children'] = {}
            child_ids = extract_child_ids_for_study(study, data['studies'][study]['reader'])
            f.write(f'    {study:<9} n={len(child_ids)}\n')
            for child_id in child_ids:
                # Initialize the child's data in the data structure
                data['studies'][study]['children'][child_id] = {'transcripts': {}}

                total += 1
        f.write(f'    ---------------\n    {"Total":<9} N={total}\n')

    return data


def determine_age_availability(data, studies):
    """
    Determine the ages at which data is available, rounding to the nearest month.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        studies (list of strings): The names of the studies (e.g. Bates, Champaign, etc.).

    Returns:
        data (dict): The input dataset updated with age availability data.
    """
    # Calculate unique ages (after rounding).
    ages_of_measurement = sorted(
        round(age_of_measurement)
        for data in data['studies'].values()
        if 'reader' in data
        for age_of_measurement in data['reader'].ages(months=True)
        if age_of_measurement not in [None, 0]
    )
    
    # Create buckets for each valid age of measurement, each containing gender-specific readers and transcripts.
    data['ages_of_measurement'] = {age_of_measurement: {
        'male':   {'readers': [], 'transcripts': []},
        'female': {'readers': [], 'transcripts': []},
        } for age_of_measurement in ages_of_measurement}

    # Plot the number of children in each age bucket.
    age_of_measurement_counter = Counter(ages_of_measurement)
    ages = [str(age) for age, _ in age_of_measurement_counter.items()]
    counts = [count for _, count in age_of_measurement_counter.items()]
    plotting.plot_age_availability(ages, counts, studies)

    return data


def organize_by_child_age_and_gender(data, studies):
    """
    Organize pylangacq.Reader objects by child age and gender.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        studies (list of strings): The names of the studies (e.g. Bates, Champaign, etc.).

    Returns:
        data (dict): The input dataset updated with pylangacq.Reader objects organized by child age and gender.
    """
    num_invalid_ages, num_valid_ages, num_invalid_genders, num_valid_genders = 0, 0, 0, 0
    data['genders'] = {'male': pylangacq.Reader(), 'female': pylangacq.Reader()}
    for study, study_data in data['studies'].items():
        for child_id, child_data in study_data['children'].items():
            child_reader = pylangacq.read_chat(f'data/{study}.zip', match=child_id)
            
            # Attempt to extract child gender
            gender = (child_reader.headers()[0].get('Participants', {}).get('CHI', {}).get('sex') or None)
            child_data['gender'] = gender
            num_valid_genders += bool(gender)
            num_invalid_genders += not bool(gender)
            if not gender:
                # No valid gender found, discard this child's data
                continue

            # Attempt to extract child age(s) (there may be multiple if the child was measured at multiple ages)
            for age_reader in child_reader:
                ages = age_reader.ages(months=True)
                if len(ages) != 1 or ages[0] in [None, 0]:
                    # No valid age found, discard this child's data
                    num_invalid_ages += 1
                    continue
                age = round(ages[0])
                num_valid_ages += 1

                # Store the reader object in the appropriate age and gender buckets
                data['ages_of_measurement'][age][gender]['readers'].append(age_reader)
                data['genders'][gender].append(age_reader)
                child_data['transcripts'].setdefault(age, []).append(age_reader)

    # Create a file for writing the number of invalid ages and genders that were discarded
    output_dir = f"results/{'_'.join(studies)}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/discard_summary.txt"
    with open(output_file, "w") as f:
        if num_valid_ages + num_invalid_ages:
            f.write(f'Discarded {num_invalid_ages} readers with invalid ages ({round(100 * num_invalid_ages / (num_valid_ages + num_invalid_ages))}%).\n')
        if num_valid_genders + num_invalid_genders:
            f.write(f'Discarded {num_invalid_genders} invalid genders ({round(100 * num_invalid_genders / (num_valid_genders + num_invalid_genders))}%).\n')

    return data


def extract_transcripts_by_age_and_gender(data, studies):
    """
    Extract transcripts by child age and gender.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        studies (list of strings): The names of the studies (e.g. Bates, Champaign, etc.).

    Returns:
        data (dict): The input dataset updated with transcripts organized according to child age and gender.
        ages_of_measurement_buckets (list): A list of the ages of measurement.
    """
    # Create a file for writing the gender distrubution for the given studies
    output_dir = f"results/{'_'.join(studies)}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/transcript_counts.txt"

    male_transcripts_count = female_transcripts_count = num_child_words = num_parent_words = 0
    for age_data in data['ages_of_measurement'].values():
        for gender in ['male', 'female']:
            # Extract transcripts
            transcripts = [
                {
                    'child': ' '.join(reader.words(participants='CHI')),
                    'parent': ' '.join(reader.words(participants=['MOT', 'FAT']))
                }
                for reader in age_data[gender]['readers']
            ]
            age_data[gender]['transcripts'] = transcripts
            
            # Record statistics
            num_child_words += sum(len(t['child'].split()) for t in transcripts)
            num_parent_words += sum(len(t['parent'].split()) for t in transcripts)
            if gender == 'male':
                male_transcripts_count += len(transcripts)
            else:
                female_transcripts_count += len(transcripts)

    # Calculate aggregated statistics
    total_transcripts = male_transcripts_count + female_transcripts_count
    percent_male = round(100 * male_transcripts_count / total_transcripts)
    percent_female = 100 - percent_male
    total_words = num_child_words + num_parent_words
    percent_child_words = round(100 * num_child_words / total_words)

    # Write statistics to the file
    with open(output_file, "w") as f:
        f.write('\nTotal number of transcripts:\n')
        f.write(f'    Male   child: {male_transcripts_count} ({percent_male}%)\n')
        f.write(f'    Female child: {female_transcripts_count} ({percent_female}%)\n')
        f.write('\nTotal number of words:\n')
        f.write(f'    Child words: {num_child_words}\n')
        f.write(f'    Parent words: {num_parent_words}\n')
        f.write(f'    Percent child words: {percent_child_words}%\n')

    ages_of_measurement_buckets = data['ages_of_measurement'].keys()
    return data, ages_of_measurement_buckets


def bucket_transcripts_by_age_of_measurement(data, ages_of_measurement_buckets, participants, studies):
    """
    Bucket transcripts by age of measurement.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        ages_of_measurement_buckets (list): A list of the ages of measurement.
        studies (list of strings): The names of the studies (e.g. Bates, Champaign, etc.).

    Returns:
        child_transcripts_by_age (dict): Transcripts organized by age of measurement.
    """
    # Create the output directory if it doesn't exist
    output_dir = f"results/{'_'.join(studies)}"
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


def organize_transcripts_by_age_of_measurement_and_child_id_for_studies(data, studies, participant):
    """
    Organize transcripts by age of measurement and child ID for a list of studies.

    Args:
        data (dict): The dataset containing study, child, and transcript information.
        studies (list): A list of studies to include in the organization.
        participant (str): The participant type ('child' or 'parent').

    Returns:
        dict: A dictionary organized by age of measurement and child ID across the specified studies.
    """
    # Collect all unique ages of measurement across the studies
    study_ages_of_measurement = set()
    for study in studies:
        for child_id in data['studies'][study]['children'].keys():
            study_ages_of_measurement.update(data['studies'][study]['children'][child_id]['transcripts'].keys())

    # Initialize structure for organizing transcripts by age
    child_transcripts_by_age_of_measurement = {age: {} for age in study_ages_of_measurement}

    # Process transcripts for each study
    for study in studies:
        for age in study_ages_of_measurement:
            for child_id, child_data in data['studies'][study]['children'].items():
                if age in child_data['transcripts']:
                    # Preprocess documents for the specified participant type
                    child_transcripts_by_age_of_measurement[age][child_id] = text_preprocessing.preprocess_documents(
                        [' '.join(reader.words(participants='CHI' if participant == 'child' else ['MOT', 'FAT']))
                         for reader in child_data['transcripts'][age]],
                        flatten=True
                    )
                else:
                    # No transcripts for this age, set an empty list
                    child_transcripts_by_age_of_measurement[age][child_id] = []

    return child_transcripts_by_age_of_measurement