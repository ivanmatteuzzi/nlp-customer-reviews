import streamlit as st
import joblib


# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="📊"
)


st.title("Amazon Customer Review Sentiment Analyzer")

st.write(
    "Enter a customer review and the model will predict its sentiment."
)


review = st.text_area(
    "Customer Review",
    placeholder="Paste a customer review here...",
    height=180
)


if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a review.")

    else:
        # Convert the review into TF-IDF features
        X = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(X)[0]

        # Get confidence scores
        probabilities = model.predict_proba(X)[0]

        scores = {
            label: float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        st.subheader("Prediction")

        if prediction == "positive":
            st.success("POSITIVE")
        elif prediction == "negative":
            st.error("NEGATIVE")
        else:
            st.warning("NEUTRAL")

        st.subheader("Confidence")

        for label, score in scores.items():
            st.write(f"**{label.capitalize()}: {score:.1%}**")

        st.bar_chart(scores)