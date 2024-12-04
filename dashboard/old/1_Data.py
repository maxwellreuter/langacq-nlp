import streamlit as st
import pandas as pd

# Data
bates_c = 56107
champaign_c = 1396883
garvey_c = 61885
hall_c = 1350935
hslld_c = 1797326

bates_c_unique = 1635
champaign_c_unique = 7090
garvey_c_unique = 2119
hall_c_unique = 13631
hslld_c_unique = 16682

# Calculations
num_words_total = bates_c + champaign_c + garvey_c + hall_c + hslld_c
num_unique_words_total = bates_c_unique + champaign_c_unique + garvey_c_unique + hall_c_unique + hslld_c_unique

average_words_per_minute_spoken_heuristic = 130
time_spoken_total = num_words_total / average_words_per_minute_spoken_heuristic
time_spoken_total_unique = num_unique_words_total / average_words_per_minute_spoken_heuristic

# Streamlit app
st.title("Word Count Analysis")

# Total Number of Words
st.subheader("Total Number of Words")
st.write(num_words_total)

# Fraction of Words by Source
st.subheader("Fraction of Words by Source")
data_fraction = {
    "Source": ["Bates", "Champaign", "Garvey", "Hall", "HSLLD"],
    "Word Count": [bates_c, champaign_c, garvey_c, hall_c, hslld_c],
    "Fraction (%)": [
        int(100 * bates_c / num_words_total),
        int(100 * champaign_c / num_words_total),
        int(100 * garvey_c / num_words_total),
        int(100 * hall_c / num_words_total),
        int(100 * hslld_c / num_words_total),
    ]
}
df_fraction = pd.DataFrame(data_fraction)
st.table(df_fraction)

# Total Number of Unique Words
st.subheader("Total Number of Unique Words")
st.write(num_unique_words_total)
st.write(f"Ratio of Unique Words (Word Diversity): {round(num_unique_words_total / num_words_total, 2)}")

# Word Diversity Ratios
st.subheader("Word Diversity Ratios")
data_diversity = {
    "Source": ["Bates", "Champaign", "Garvey", "Hall", "HSLLD"],
    "Unique Word Count": [bates_c_unique, champaign_c_unique, garvey_c_unique, hall_c_unique, hslld_c_unique],
    "Total Word Count": [bates_c, champaign_c, garvey_c, hall_c, hslld_c],
    "Diversity Ratio": [
        round(bates_c_unique / bates_c, 3),
        round(champaign_c_unique / champaign_c, 3),
        round(garvey_c_unique / garvey_c, 3),
        round(hall_c_unique / hall_c, 3),
        round(hslld_c_unique / hslld_c, 3),
    ]
}
df_diversity = pd.DataFrame(data_diversity)
st.table(df_diversity)

# Total Time Spoken
st.subheader("Total Time Spoken")
st.write(f"{round(time_spoken_total/(60*24), 1)} days ({round(time_spoken_total/(60))} hours)")
st.write(f"Total Time Spoken (Unique Words): {round(time_spoken_total_unique/(60*24), 1)} days ({round(time_spoken_total_unique/(60))} hours)")
# Hours Spent by Source
st.subheader("Hours Spent by Source")
data_hours = {
    "Source": ["Bates", "Champaign", "Garvey", "Hall", "HSLLD"],
    "Hours": [
        int(bates_c / average_words_per_minute_spoken_heuristic / 60),
        int(champaign_c / average_words_per_minute_spoken_heuristic / 60),
        int(garvey_c / average_words_per_minute_spoken_heuristic / 60),
        int(hall_c / average_words_per_minute_spoken_heuristic / 60),
        int(hslld_c / average_words_per_minute_spoken_heuristic / 60),
    ]
}
df_hours = pd.DataFrame(data_hours)
st.table(df_hours)


st.subheader("POS Frequency as a Function of Child Age")
st.image("img/pos.png", caption="POS Frequency as a Function of Child Age", use_column_width=True)


