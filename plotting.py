import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import string
import numpy as np

import helpers

def plot_age_availability(ages, counts, study):
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
    plt.savefig(f"results/{study}/ages.png")
    plt.close()

def plot_venn_diagram(group_a_unique, group_b_unique, group_a_b_common, group_a, group_b, group_a_unique_str, group_b_unique_str, group_a_b_common_str, study):
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
    plt.savefig(f"results/{study}/ngrams_{group_a}_{group_b}.png")
    plt.close()

def plot_pca(explained_variance, participant, study):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), explained_variance[:8], marker='o', label="Eigenvalues")
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Scree Plot: Explained Variance by Principal Components ({participant}')
    plt.legend()
    plt.grid()
    plt.savefig(f"results/{study}/pca_{participant}.png")
    plt.close()

def plot_lda(smoothed, participant, study, window_size):
    plt.figure(figsize=(12, 8))
    for column in smoothed.columns:
        plt.plot(smoothed.index, smoothed[column], label=column)
    plt.xlabel("Age in Months")
    plt.ylabel("Smoothed Topic Proportion")
    plt.title(f"Smoothed Topic Proportions Across Ages (Window Size = {window_size}) ({participant})")
    plt.legend()
    plt.savefig(f"results/{study}/topics_{participant}.png")
    plt.close()

def plot_simplex(all_points, all_colors, all_labels, corners, topic_labels, participant, study, age):
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
    plt.savefig(f"results/{study}/clusters/{participant}/age_{age}.png")
    plt.close()

def plot_ngrams(data, study):
    """
    Analyze n-grams for male vs. female participants and generate Venn diagrams.
    """
    genders = ['male', 'female']
    participants = {'child': ['CHI'], 'parent': ['MOT', 'FAT']}

    # Calculate n-grams
    def calculate_ngrams(n):
        return {
            gender: {
                role: Counter({
                    ngram: count
                    for ngram, count in data['genders'][gender].word_ngrams(n=n, participants=roles).items()
                    if all(word not in string.punctuation for word in ngram)
                })
                for role, roles in participants.items()
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
                                   group_a_str, group_b_str, group_common_str, study)

    # Perform calculations and generate plots
    n = 3
    ngrams = calculate_ngrams(n)
    print(f'Completed calculations for {n}-grams.')

    if study not in ['All', 'Garvey']:
        generate_venn(
            Counter(ngrams['male']['parent']) + Counter(ngrams['female']['parent']),
            Counter(ngrams['male']['child']) + Counter(ngrams['female']['child']),
            'Parents', 'Children'
        )
    generate_venn(
        Counter(ngrams['male']['child']),
        Counter(ngrams['female']['child']),
        'Boys', 'Girls'
    )

def plot_participant_topics_on_simplex_with_tracking(lda, vectorizer, participant_transcripts, most_recent_locations, age, topic_labels, study, data, participant):
    def barycentric_coordinates(topic_distributions):
        """
        Map topic distributions into 2D barycentric coordinates.
        Place the simplex corners correctly for 4 topics in a rotated square layout.
        """
        # Define the 2D coordinates for the corners of the rotated square
        corners = np.array([
            [-1, 1],  # Topic 1 (Top-left)
            [1, 1],   # Topic 2 (Top-right)
            [1, -1],  # Topic 3 (Bottom-right)
            [-1, -1]  # Topic 4 (Bottom-left)
        ])

        # Normalize the topic distributions to sum to 1
        topic_distributions = topic_distributions / topic_distributions.sum(axis=1, keepdims=True)

        # Compute the weighted sum of the simplex corners based on topic probabilities
        points = np.dot(topic_distributions, corners)
        return points, corners
    
    """
    Predict topics for participants and plot them on a 2D simplex.
    Participants without transcripts in the current age are plotted with muted colors.
    Participants with 'G' in their name are pink, and those with 'B' are blue.
    The corners of the simplex are labeled using meaningful topic labels.
    """
    # Extract transcripts and their IDs
    participants_with_transcripts = list(participant_transcripts.keys())
    transcripts = list(participant_transcripts.values())

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
        gender = data['studies'][study]['children'][participant_id]['gender']
        if gender == 'female':
            color = 'pink'
        elif gender == 'male':
            color = 'blue'
        else:
            color = 'green'  # Default color for participants without G or B
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
            gender = data['studies'][study]['children'][participant_id]['gender']
            if gender == 'female':
                color = 'lightpink'  # Muted pink
            elif gender == 'male':
                color = 'lightblue'  # Muted blue
            else:
                color = 'lightgreen'  # Default muted color
            all_points.append(missing_points[i])
            all_labels.append(participant_id)
            all_colors.append(color)

    # Convert all points to a NumPy array for plotting
    all_points = np.array(all_points)

    plot_simplex(all_points, all_colors, all_labels, corners, topic_labels, participant, study, age)
