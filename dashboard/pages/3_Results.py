import streamlit as st
import pandas as pd

Bates = [20, 22, 28]
Champaign = [20, 21, 22, 23, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37]
Garvey = [34, 35, 36, 37, 38, 39, 42, 43, 45, 48, 49, 53, 55, 57, 58, 60, 61, 62, 63, 64, 67]
Hall = [54, 57]
HSLLD = [
    43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
    57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
    71,  72,  73,  74,  76,  77,  78,  83,  84,  85,  86,  87,  88,  89,
    90,  91,  92,  93,  95, 101, 107, 108, 109, 110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 131,
    132, 133, 138, 140, 142, 144
]
All = [
    20,  21,  22,  23,  24,  25,  27,  28,  30,  31,  33,  34,  35,  36,
    37,  38,  39,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
    53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
    67,  68,  69,  70,  71,  72,  73,  74,  76,  77,  78,  83,  84,  85,
    86,  87,  88,  89,  90,  91,  92,  93,  95, 101, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    125, 126, 127, 131, 132, 133, 138, 140, 142, 144
]

studies = {
    "All": All,
    "Bates": Bates,
    "Champaign": Champaign,
    "Garvey": Garvey,
    "Hall": Hall,
    "HSLLD": HSLLD,
}

ws = {
    "All": {
        "child": "31",
        "parent": "31" 
    },
    "Bates": {
        "child": "1",
        "parent": "1" 
    },
    "Champaign": {
        "child": "5",
        "parent": "5" 
    },
    "Garvey": {
        "child": "7",
        "parent": "7 " 
    },
    "Hall": {
        "child": "0",
        "parent": "0" 
    },
    "HSLLD": {
        "child": "25",
        "parent": "25" 
    }
}

# Define the data
data = {
    "All": {
        "transcripts": {
            "male": 842,
            "female": 2083,
        },
        "words": {
            "child": 1579313,
            "parent": 2559289
        },
        "ages": {
            "age": ['20', '21', '22', '23', '24', '25', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '76', '77', '78', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '95', '101', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '131', '132', '133', '138', '140', '142', '144'],
            "transcripts": [28, 73, 11, 2, 79, 8, 78, 82, 68, 17, 1, 70, 14, 2, 54, 30, 2, 1, 4, 21, 21, 36, 55, 33, 56, 45, 25, 25, 4, 26, 13, 32, 25, 77, 26, 25, 23, 31, 18, 9, 36, 30, 30, 56, 55, 33, 25, 36, 17, 10, 4, 1, 1, 2, 3, 6, 14, 17, 18, 22, 32, 20, 17, 20, 4, 2, 2, 3, 3, 5, 8, 11, 8, 10, 9, 8, 14, 10, 11, 3, 3, 13, 12, 1, 4, 3, 2, 3, 1, 1, 1, 1, 1, 2, 1]
        },
        "gender": {
            "male": "47%",
            "female": "40%",
            "unsp": "13%"
        },
        "pca": {
            "child": [0.17134255, 0.07910432, 0.06104795, 0.0396524, 0.03415983, 0.03120775, 0.02875815, 0.02508611],
            "parent": [0.17315044, 0.05814859, 0.05053763, 0.03775561, 0.03439426, 0.03018443, 0.02937335, 0.02556794]
        }
    },
    "Bates": {
        "transcripts": {
            "male": 40,
            "female": 44,
        },
        "words": {
            "child": 10861,
            "parent": 34732
        },
        "ages": {
            "age": ['20', '22', '28'],
            "transcripts": [27, 1, 72]
        },
        "gender": {
            "male": "48%",
            "female": "52%",
            "unsp": "0"
        },
        "pca": {
            "child": [0.17134255, 0.07910432, 0.06104795, 0.0396524, 0.03415983, 0.03120775, 0.02875815, 0.02508611],
            "parent": [0.17315044, 0.05814859, 0.05053763, 0.03775561, 0.03439426, 0.03018443, 0.02937335, 0.02556794]
        }
    },
    "Champaign": {
        "transcripts": {
            "male": 202,
            "female": 168
        },
        "words": {
            "child": 299986,
            "parent": 503937
        },
        "ages": {
            "age": ['20', '21', '22', '23', '24', '25', '27', '28', '30', '31', '32', '33', '34', '36', '37'],
            "transcripts": [1, 73, 10, 2, 79, 8, 78, 10, 68, 17, 1, 70, 12, 53, 26]
        },
        "gender": {
            "male": "42%",
            "female": "35%",
            "unsp": "22%"
        },
        "pca": {
            "child": [0.10953319, 0.07184249, 0.03679039, 0.03596078, 0.03007334, 0.02303825, 0.01913307, 0.01761913],
            "parent": [0.15225318, 0.05381482, 0.04261236, 0.03371845, 0.03302722, 0.02663543, 0.02545507, 0.02044268]
        }
    },
    "Garvey": {
        "transcripts": {
            "male": 24,
            "female": 21
        },
        "words": {
            "child": 30170,
            "parent": 0
        },
        "ages": {
            "age": ['34', '35', '36', '37', '38', '39', '42', '43', '45', '48', '49', '53', '55', '57', '58', '60', '61', '62', '63', '64', '67'],
            "transcripts": [2, 2, 1, 4, 2, 1, 4, 1, 1, 3, 1, 2, 1, 3, 1, 2, 6, 3, 2, 1, 2]
        },
        "gender": {
            "male": "53%",
            "female": "47%",
            "unsp": "0%"
        },
        "pca": {
            "child": [0.10124855, 0.0814149,  0.05861131, 0.05434228, 0.04431017, 0.04152925, 0.0408812, 0.03527427],
            "parent": [0.15225318, 0.05381482, 0.04261236, 0.03371845, 0.03302722, 0.02663543, 0.02545507, 0.02044268]
        }
    },
    "Hall": {
        "transcripts": {
            "male": 26,
            "female": 12,
        },
        "words": {
            "child": 378054,
            "parent": 300537
        },
        "ages": {
            "age": ['54', '57'],
            "transcripts": [1, 38]
        },
        "gender": {
            "male": "48%",
            "female": "52%",
            "unsp": "0%"
        },
        "pca": {
            "child": [0.18926899, 0.11752181, 0.06481014, 0.04914909, 0.04037094, 0.03963472, 0.03645308, 0.03158099],
            "parent": [0.35337788, 0.06373437, 0.05359791, 0.04230719, 0.03955285, 0.03487106, 0.03194174, 0.0299678 ]
        }
    },
    "HSLLD": {
        "transcripts": {
            "male": 550,
            "female": 1838,
        },
        "words": {
            "child": 860242,
            "parent": 1720083
        },
        "ages": {
            "age": ['43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '76', '77', '78', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '95', '101', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '131', '132', '133', '138', '140', '142', '144'],
            "transcripts": [20, 21, 35, 55, 33, 53, 44, 25, 25, 4, 24, 12, 31, 25, 36, 25, 25, 21, 25, 15, 7, 35, 30, 30, 54, 55, 33, 25, 36, 17, 10, 4, 1, 1, 2, 3, 6, 14, 17, 18, 22, 32, 20, 17, 20, 4, 2, 2, 3, 3, 5, 8, 11, 8, 10, 9, 8, 14, 10, 11, 3, 3, 13, 12, 1, 4, 3, 2, 3, 1, 1, 1, 1, 1, 2, 1]
        },
        "gender": {
            "male": "39%",
            "female": "39%",
            "unsp": "22%"
        },
        "pca": {
            "child": [0.09143637, 0.06498997, 0.02274044, 0.02244211, 0.01942599, 0.01529737, 0.01435175, 0.01286299],
            "parent": [0.1629043, 0.10538795, 0.04241368, 0.02698153, 0.02630575, 0.01956128, 0.01700747, 0.01568745]
        }
    },
}

st.title("Results")
study = st.selectbox("Choose Study", ['All', 'Bates', 'Champaign', 'Garvey', 'Hall', 'HSLLD'])
ages = studies[study]

# Calculate percentages
total_transcripts = data[study]["transcripts"]["male"] + data[study]["transcripts"]["female"]
percent_male = (data[study]["transcripts"]["male"] / total_transcripts) * 100
percent_female = (data[study]["transcripts"]["female"] / total_transcripts) * 100

total_words = data[study]["words"]["child"] + data[study]["words"]["parent"]
percent_child_words = (data[study]["words"]["child"] / total_words) * 100

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transcripts")
    st.dataframe([
        {"": "Male", "Count": data[study]["transcripts"]["male"], "Percentage": f"{percent_male:.0f}%"},
        {"": "Female", "Count": data[study]["transcripts"]["female"], "Percentage": f"{percent_female:.0f}%"},
    ], use_container_width=True)

with col2:
    st.subheader("Words")
    st.dataframe([
        {"": "Child", "Count": data[study]["words"]["child"], "Percentage": f"{percent_child_words:.0f}%"},
        {"": "Parent", "Count": data[study]["words"]["parent"], "Percentage": f"{100 - percent_child_words:.0f}%"}
    ], use_container_width=True)


st.subheader("Gender")
st.dataframe([
    {"": "Male", "Percentage": data[study]["gender"]["male"]},
    {"": "Female", "Percentage": data[study]["gender"]["female"]},
    {"": "Unspecified", "Percentage": data[study]["gender"]["unsp"]},
], use_container_width=True)



st.subheader("Transcript Counts by Age")
df = pd.DataFrame(data[study]["ages"])
df = df.set_index("age")
st.bar_chart(df, x_label="Age (months)", y_label="Number of Transcripts")




smoothed_child = pd.read_csv(f"results/{study}/child.csv")
smoothed_child.index = index=[int(x) for x in ages]

st.subheader("Child - Smoothed Topic Proportions Across Ages")
st.write(f'(window size = {ws[study]["child"]})')
st.line_chart(smoothed_child, x_label='Ages in Months', y_label='Smoothed Topic Proportion')

st.subheader("Child - Explained Variance by Principal Component")
st.line_chart(data[study]["pca"]["child"], x_label="Principal Component", y_label="Explained Variance Ratio")


if study != "Garvey":
    smoothed_parent = pd.read_csv(f"results/{study}/parent.csv")
    smoothed_parent.index = index=[int(x) for x in ages]
    st.subheader("Parent - Smoothed Topic Proportions Across Ages")
    st.write(f'(window size = {ws[study]["parent"]})')
    st.line_chart(smoothed_parent, x_label='Ages in Months', y_label='Smoothed Topic Proportion')

    st.subheader("Parent - Explained Variance by Principal Component")
    st.line_chart(data[study]["pca"]["child"], x_label="Principal Component", y_label="Explained Variance Ratio")


st.subheader("Top N-grams")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Boys and Girls")
    st.image(f"img/results/{study}/ngrams_Boys_Girls.png", use_column_width=True)

with col2:
    if study != "Garvey":
        st.subheader("Parents and Children")
        st.image(f"img/results/{study}/ngrams_Parents_Children.png", use_column_width=True)
   





