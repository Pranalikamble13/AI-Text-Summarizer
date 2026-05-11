import streamlit as st
import nltk

# ---------------- NLTK ----------------

nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------- EXTRACTIVE ----------------

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# ---------------- ABSTRACTIVE ----------------

from transformers import pipeline

# ---------------- EVALUATION ----------------

from rouge_score import rouge_scorer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="centered"
)

# ---------------- TITLE ----------------

st.title("📝 AI Text Summarization System")

st.write("Perform Extractive and Abstractive Summarization with Evaluation.")

# ---------------- LOAD ABSTRACTIVE MODEL ----------------

@st.cache_resource
def load_model():

    summarizer = pipeline(
        "summarization",
        model="t5-small"
    )

    return summarizer

abstractive_model = load_model()

# ---------------- USER INPUT ----------------

text = st.text_area(
    "Enter Long Text",
    height=300
)

# ---------------- SUMMARY TYPE ----------------

summary_type = st.radio(
    "Choose Summary Type",
    [
        "Extractive Summarization",
        "Abstractive Summarization"
    ]
)

# ---------------- SENTENCE SLIDER ----------------

num_sentences = st.slider(
    "Number of Sentences (Extractive)",
    1,
    10,
    3
)

# ---------------- BUTTON ----------------

if st.button("Generate Summary"):

    if text.strip() == "":

        st.warning("Please enter text.")

    else:

        # ====================================================
        # EXTRACTIVE SUMMARIZATION
        # ====================================================

        if summary_type == "Extractive Summarization":

            with st.spinner("Generating Extractive Summary..."):

                parser = PlaintextParser.from_string(
                    text,
                    Tokenizer("english")
                )

                summarizer = LexRankSummarizer()

                summary = summarizer(
                    parser.document,
                    num_sentences
                )

                final_summary = ""

                for sentence in summary:
                    final_summary += str(sentence) + " "

                st.subheader("Extractive Summary")

                st.success(final_summary)

                # ---------------- EVALUATION ----------------

                scorer = rouge_scorer.RougeScorer(
                    ['rouge1'],
                    use_stemmer=True
                )

                scores = scorer.score(text, final_summary)

                st.subheader("Evaluation")

                st.write("### ROUGE-1 Score")

                st.write(
                    "Precision:",
                    round(scores['rouge1'].precision, 2)
                )

                st.write(
                    "Recall:",
                    round(scores['rouge1'].recall, 2)
                )

                st.write(
                    "F1 Score:",
                    round(scores['rouge1'].fmeasure, 2)
                )

        # ====================================================
        # ABSTRACTIVE SUMMARIZATION
        # ====================================================

        elif summary_type == "Abstractive Summarization":

            with st.spinner("Generating Abstractive Summary..."):

                # Reduce very long text
                text = text[:1000]

                input_text = "summarize: " + text

                result = abstractive_model(
                    input_text,
                    max_length=80,
                    min_length=20,
                    do_sample=False
                )

                final_summary = result[0]['summary_text']

                st.subheader("Abstractive Summary")

                st.success(final_summary)

                # ---------------- EVALUATION ----------------

                scorer = rouge_scorer.RougeScorer(
                    ['rouge1'],
                    use_stemmer=True
                )

                scores = scorer.score(text, final_summary)

                st.subheader("Evaluation")

                st.write("### ROUGE-1 Score")

                st.write(
                    "Precision:",
                    round(scores['rouge1'].precision, 2)
                )

                st.write(
                    "Recall:",
                    round(scores['rouge1'].recall, 2)
                )

                st.write(
                    "F1 Score:",
                    round(scores['rouge1'].fmeasure, 2)
                )