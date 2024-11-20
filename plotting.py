import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import string
import numpy as np
import os

import helpers

def plot_age_availability(ages, counts, studies):
    plt.figure(figsize=(14, 6))  # Increase figure size for clarity
    plt.bar(ages, counts, color='skyblue')  # Use a visually appealing color

    # Customize the plot
    plt.xticks(rotation=45, ha='right')  # Rotate x-tick labels for better legibility
    plt.xlabel('Age of Measurement (months)')
    plt.ylabel('Number of Transcripts')
    plt.title('Number of Transcripts at Each Age of Measurement (months)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add horizontal grid lines

    # Adjust the layout
    plt.tight_layout()  # Prevent x-tick labels from being cut off

    #plt.show()
    save_path = f"results/{'_'.join(studies)}/ages.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_venn_diagram(group_a_unique, group_b_unique, group_a_b_common, group_a, group_b, group_a_unique_str, group_b_unique_str, group_a_b_common_str, studies):
    # Plot the Venn diagram
    plt.figure(figsize=(10, 10))
    venn = venn3(
        subsets=(len(group_a_unique), len(group_b_unique), len(group_a_b_common), 0, 0, 0, 0),
        set_labels=(group_a, group_b)
    )

    # Customize the Venn diagram with examples
    venn.get_label_by_id('100').set_text(f"{group_a_unique_str}")
    venn.get_label_by_id('010').set_text(f"{group_b_unique_str}")
    venn.get_label_by_id('110').set_text(f"{group_a_b_common_str}")

    plt.title(f'Top N-grams Among {group_a} and {group_b}')
    save_path = f"results/{'_'.join(studies)}/ngrams_{group_a}_{group_b}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_pca(explained_variance, participant, studies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), explained_variance[:8], marker='o', label="Eigenvalues")
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Scree Plot: Explained Variance by Principal Components ({participant})')
    plt.legend()
    plt.grid()
    save_path = f"results/{'_'.join(studies)}/pca_{participant}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_lda(smoothed, participant, studies, window_size):
    plt.figure(figsize=(12, 8))
    for column in smoothed.columns:
        plt.plot(smoothed.index, smoothed[column], label=column)
    plt.xlabel("Age in Months")
    plt.ylabel("Smoothed Topic Proportion")
    plt.title(f"Smoothed Topic Proportions Across Ages (Window Size = {window_size}) ({participant})")
    plt.legend()
    save_path = f"results/{'_'.join(studies)}/topics_{participant}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_simplex(all_points, all_colors, all_labels, corners, topic_labels, participant, studies, age):
    # Plot the simplex
    plt.figure(figsize=(12, 10))
    plt.scatter(all_points[:, 0], all_points[:, 1], c=all_colors, s=100, alpha=0.7)

    # Annotate points with participant IDs
    for i, participant_id in enumerate(all_labels):
        plt.annotate(participant_id, (all_points[i, 0], all_points[i, 1]), fontsize=10, ha='center', va='center')

    # Plot simplex corners with meaningful labels
    for i, (x, y) in enumerate(corners):
        plt.scatter(x, y, c='red', s=200, label=topic_labels[i])  # Use the passed topic labels
        plt.annotate(topic_labels[i], (x, y), fontsize=12, fontweight='bold', ha='center', va='center')

    # Add grid, title, and labels
    plt.title(f"2D Representation of Participant Topic Distributions (Age: {age}) ({participant})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(alpha=0.5)

    # Save the plot as a PNG file
    save_path = f"results/{'_'.join(studies)}/clusters/{participant}/age_{age}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_ngrams(data, studies):
    """
    Analyze n-grams for male vs. female participants and generate Venn diagrams.
    """
    genders = ['male', 'female']
    participants = {'child': ['CHI'], 'parent': ['MOT', 'FAT']}

    # Calculate n-grams
    def calculate_ngrams(n, studies):
        results = Counter()
        for study in studies:
            for gender in genders:
                for role, roles in participants.items():
                    if study == 'Garvey' and role == 'parent':
                        print(f'Skipping parent figures for Garvey during n-gram calculations for {gender} children.')
                        continue
                    
                    for ngram, count in data['genders'][gender].word_ngrams(n=n, participants=roles).items():
                        if all(word not in string.punctuation for word in ngram):
                            results[(gender, role, ngram)] += count

        return {
            gender: {
                role: Counter({
                    ngram: results[(gender, role, ngram)]
                    for (g, r, ngram) in results.keys()
                    if g == gender and r == role
                })
                for role in participants.keys()
            }
            for gender in genders
        }

    # Generate Venn diagram for two groups
    def generate_venn(group_a_counter, group_b_counter, label_a, label_b):
        top_a = set(ngram for ngram, _ in group_a_counter.most_common(40))
        top_b = set(ngram for ngram, _ in group_b_counter.most_common(40))

        group_a_unique = [ngram for ngram in group_a_counter if ngram not in top_b]
        group_b_unique = [ngram for ngram in group_b_counter if ngram not in top_a]
        group_common = set(group_a_counter) & set(group_b_counter)

        def top_ngrams(ngrams, counter, limit=10):
            return sorted(ngrams, key=lambda x: counter[x], reverse=True)[:limit]

        group_a_str = "\n".join([" ".join(ngram) for ngram in top_ngrams(group_a_unique, group_a_counter)])
        group_b_str = "\n".join([" ".join(ngram) for ngram in top_ngrams(group_b_unique, group_b_counter)])
        group_common_str = "\n".join([" ".join(ngram) for ngram in top_ngrams(group_common, group_a_counter)])

        plot_venn_diagram(group_a_unique, group_b_unique, group_common, label_a, label_b,
                                   group_a_str, group_b_str, group_common_str, studies)

    # Perform calculations and generate plots
    n = 3
    ngrams = calculate_ngrams(n, studies)
    print(f'Completed calculations for {n}-grams.')

    try:
        generate_venn(
            Counter(ngrams['male']['parent']) + Counter(ngrams['female']['parent']),
            Counter(ngrams['male']['child']) + Counter(ngrams['female']['child']),
            'Parents', 'Children'
        )
    except:
        print('Skipping parent figures for Garvey during Venn diagram generation.')
    generate_venn(
        Counter(ngrams['male']['child']),
        Counter(ngrams['female']['child']),
        'Boys', 'Girls'
    )

def plot_participant_topics_on_simplex_with_tracking(
    lda, vectorizer, participant_transcripts, most_recent_locations, age, topic_labels, studies, data, participant
):
    """
    Predict topics for participants across multiple studies and plot them on a 2D simplex.
    """

    def barycentric_coordinates(topic_distributions):
        # Map topic distributions into 2D barycentric coordinates.
        corners = np.array([
            [-1, 1],  # Topic 1 (Top-left)
            [1, 1],   # Topic 2 (Top-right)
            [1, -1],  # Topic 3 (Bottom-right)
            [-1, -1]  # Topic 4 (Bottom-left)
        ])
        topic_distributions = topic_distributions / topic_distributions.sum(axis=1, keepdims=True)
        points = np.dot(topic_distributions, corners)
        return points, corners

    # Filter participant_transcripts to exclude 'parent' for 'Garvey'
    filtered_transcripts = {
        pid: transcript
        for pid, transcript in participant_transcripts.items()
        if not (participant == 'parent' and 'Garvey' in studies)
    }

    # Extract participants and their transcripts across all studies
    participants_with_transcripts = list(filtered_transcripts.keys())
    transcripts = list(filtered_transcripts.values())

    # Check if transcripts are empty and handle gracefully
    if not transcripts:
        return

    # Predict topic distributions for participants with transcripts
    _, topic_distributions = helpers.predict_topic(lda, vectorizer, transcripts)

    # Update most recent locations for participants with transcripts
    for i, participant_id in enumerate(participants_with_transcripts):
        most_recent_locations[participant_id] = topic_distributions[i]

    # Combine points for participants with and without transcripts
    all_points = []
    all_labels = []
    all_colors = []

    # Participants with transcripts
    points_with_transcripts, corners = barycentric_coordinates(topic_distributions)
    for i, participant_id in enumerate(participants_with_transcripts):
        gender = None
        for study in studies:
            if participant_id in data['studies'][study]['children']:
                gender = data['studies'][study]['children'][participant_id]['gender']
                break
        color = 'pink' if gender == 'female' else 'blue' if gender == 'male' else 'green'
        all_points.append(points_with_transcripts[i])
        all_labels.append(participant_id)
        all_colors.append(color)

    # Participants without transcripts
    participants_without_transcripts = [
        participant_id for participant_id in most_recent_locations.keys() if participant_id not in participants_with_transcripts
    ]
    if participants_without_transcripts:
        missing_points = np.array([most_recent_locations[participant_id] for participant_id in participants_without_transcripts])
        missing_points, _ = barycentric_coordinates(missing_points)
        for i, participant_id in enumerate(participants_without_transcripts):
            gender = None
            for study in studies:
                if participant_id in data['studies'][study]['children']:
                    gender = data['studies'][study]['children'][participant_id]['gender']
                    break
            color = 'lightpink' if gender == 'female' else 'lightblue' if gender == 'male' else 'lightgreen'
            all_points.append(missing_points[i])
            all_labels.append(participant_id)
            all_colors.append(color)

    # Convert all points to a NumPy array for plotting
    all_points = np.array(all_points)

    # Plot the points on the simplex
    plot_simplex(all_points, all_colors, all_labels, corners, topic_labels, participant, studies, age)
