# README

## How to run the program and generate our results

1. The dependencies can be found in `environment.yml`.
2. The conversational topics extracted by ChatGPT will default to `NO_OPENAI_KEY_PROVIDED` unless `analysis.py` is populated with an OpenAI API key. The effects of allowing this to default are minimial besides not being able to reproduce the actual topics in some of the graphs. 
    ```python
    client = OpenAI(api_key="key_goes_here")
    ```
3. Run `python main.py`. This code takes a substantial amount of time to run (10-20+ minutes), especially because it analyzes the corpora both individually and combined.
 - The warning `UserWarning: Circle C has zero area.` can be ignored.
 - Messages like `Skipping parent figures...` can be ignored.
4. Results will be stored in the `results/` folder in this directory.

## Code organization

- `data_organization.py` mainly handles the reading and restructuring of the data (e.g. organizing it by child age instead of by study).
- `text_preprocessing.py` handles text preprocessing, such as removing stopwords and common uterrances in child-parent interactions.
- `analysis.py` handles mostly machine learning responsibilities, like PCA, LDA, etc.
- `plotting.py` handles plotting various data or information using `matplotlib`.

Except for `plotting.py`, all functions have a "header" that describes what the function does and provides details about its inputs and outputs.