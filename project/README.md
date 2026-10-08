NLP Automated Customer Reviews

Project Overview

This project focuses on the analysis of Amazon customer reviews using Natural Language Processing (NLP) and Machine Learning techniques.

The main objective was to extract useful information from customer reviews and transform unstructured textual data into actionable insights. The project was developed through four main tasks: sentiment analysis, product clustering, Generative AI-based category summaries, and deployment of a sentiment classification web application.

The project combines traditional Machine Learning techniques, unsupervised learning, and Generative AI. The final sentiment classifier was deployed as a public Streamlit web application, allowing users to enter a customer review and receive a predicted sentiment together with the model's confidence scores.

The complete analysis, experiments, models, and results are documented in the accompanying Jupyter Notebook.

## Task 1 — Sentiment Analysis

The first task focused on classifying Amazon customer reviews into three sentiment categories: **positive, neutral, and negative**.

### Data Preparation and EDA

The dataset was first explored through Exploratory Data Analysis (EDA) to understand its structure, missing values, review ratings, and sentiment distribution.

Rows with missing ratings or review text were removed before building the classification dataset. The original star ratings were then mapped into three sentiment classes:

* **Positive:** ratings 4–5
* **Neutral:** rating 3
* **Negative:** ratings 1–2

The resulting dataset contained:

* **32,316 positive reviews**
* **1,499 neutral reviews**
* **812 negative reviews**

This distribution revealed a strong class imbalance, with positive reviews representing the vast majority of the dataset.

The data was split into **80% training data and 20% test data**. The split preserved the same general class distribution:

**Training set**

* Positive: 25,852
* Neutral: 1,199
* Negative: 650

**Test set**

* Positive: 6,464
* Neutral: 300
* Negative: 162

A confusion matrix was also used to visualize the classification errors and better understand how the model performed across the three sentiment classes.

### Model Experiments

Several approaches were tested to identify a suitable baseline model.

#### Bag of Words

A Bag of Words representation was first used to convert the review text into numerical features.

The resulting model achieved approximately:

* **Accuracy: 91.4%**
* **Macro F1-score: 0.59**

#### TF-IDF + Logistic Regression

A TF-IDF representation was then combined with a Logistic Regression classifier.

The model achieved approximately:

* **Accuracy: 92.0%**
* **Macro F1-score: 0.60**

Class weighting was used to partially compensate for the strong imbalance between positive, neutral, and negative reviews. The `C` parameter was also optimized to improve the model compared with the initial baseline.

TF-IDF performed slightly better than Bag of Words, suggesting that weighting terms according to their importance within the corpus was more effective for this dataset.

#### DistilBERT

A transformer-based approach using DistilBERT was also tested.

Although DistilBERT achieved slightly better results than the simpler approaches, the improvement was not substantial enough to justify its significantly higher computational complexity and resource requirements for this task.

For this reason, the TF-IDF + Logistic Regression approach was selected as the final baseline model.

### Model Limitations

The main challenge was the severe class imbalance in the dataset. The model performed very well on the majority **positive** class, while its performance on **neutral** and **negative** reviews was considerably weaker.

Many minority-class reviews were incorrectly classified as positive. This means that accuracy alone can be misleading: a high accuracy score is partly a consequence of the large number of positive reviews.

Another limitation comes from the conversion of star ratings into sentiment categories. Grouping ratings 1–2 as negative and 4–5 as positive creates some unavoidable ambiguity. In particular, a four-star review may contain language that is closer to neutral or mixed sentiment, even though it is classified as positive according to the predefined mapping.

Therefore, **macro F1-score** was considered alongside accuracy to obtain a more meaningful evaluation of performance across all three classes.

### Final Result

The final TF-IDF + Logistic Regression model provides a strong and interpretable baseline for sentiment classification on this dataset.

It achieved approximately **92% accuracy and a 0.60 macro F1-score**, while remaining substantially simpler and less computationally demanding than the transformer-based alternative.

The model was subsequently saved and used as the basis for the public Streamlit sentiment analysis application developed in Task 4.

## Task 2 — Product Category Clustering

The second task focused on grouping the **34,626 customer reviews** into a smaller number of meaningful product meta-categories using unsupervised Machine Learning.

### Data Exploration

The original `categories` field contained a very large number of heterogeneous and highly detailed category combinations. Many rows included long lists of overlapping categories, marketplaces, departments, accessories, and product-related keywords.

For example, the same product could simultaneously appear under categories such as *Fire Tablets*, *Tablets*, *Computers & Tablets*, *Electronics*, *All Tablets*, and several other nested categories.

This made the original category field difficult to use directly for a high-level product analysis.

For this reason, instead of relying on the original categories, an unsupervised clustering approach was used to identify broader and more meaningful product groups.

### Feature Construction

A combined text field was created for each review using:

* product name
* brand
* original product categories
* review title
* review text

These textual features were combined into a single representation so that the clustering algorithm could use both **product information** and **customer review content**.

The resulting text was converted into numerical features using **TF-IDF**, with a maximum of **5,000 features**.

### K-Means Clustering

K-Means was selected as the clustering algorithm.

Different numbers of clusters were tested in order to identify a solution that provided both reasonable statistical separation and meaningful product categories.

Initially, a **5-cluster solution** was tested. The resulting clusters contained:

* Cluster 0: 3,207 reviews
* Cluster 1: 8,812 reviews
* Cluster 2: 7,261 reviews
* Cluster 3: 5,052 reviews
* Cluster 4: 10,294 reviews

Although this solution produced several recognizable groups, there was noticeable overlap between some clusters. In particular, products and reviews related to tablets, Kindle devices, and other Amazon electronics were not always clearly separated.

The clustering process was therefore evaluated using the **Silhouette Score** for 4, 5, and 6 clusters.

| Number of clusters | Silhouette Score |
| ------------------ | ---------------: |
| 4                  |           0.2024 |
| 5                  |           0.1979 |
| 6                  |           0.1963 |

The **4-cluster solution** was selected because it achieved the highest Silhouette Score and also produced the most interpretable grouping.

### Final Meta-Categories

The four resulting clusters were interpreted by examining their dominant keywords, representative products, categories, and sample reviews.

| Cluster | Meta-category      | Main topics                                            |
| ------- | ------------------ | ------------------------------------------------------ |
| 0       | E-readers & Kindle | Kindle, Paperwhite, ebooks, readers, books             |
| 1       | Smart Home & Alexa | Echo, Alexa, smart home, speakers, automation          |
| 2       | Tablets            | Fire Tablet, Fire HD, tablets, displays, Android, apps |
| 3       | Streaming & TV     | Fire TV, streaming, movies, TV, media, video           |

The final distribution was:

* **Tablets:** 17,623 reviews
* **Smart Home & Alexa:** 7,262 reviews
* **E-readers & Kindle:** 4,685 reviews
* **Streaming & TV:** 5,056 reviews

A bar chart was also created to visualize the distribution of reviews across the four final meta-categories.

### Interpretation

The clustering produced four broad and interpretable product groups that are more useful for the subsequent analysis than the original highly fragmented category field.

The separation between clusters was **limited but meaningful**, as reflected by the relatively modest Silhouette Scores. This is expected given the strong overlap between Amazon product categories and the fact that many products share similar vocabulary, particularly within the electronics ecosystem.

The resulting meta-categories were therefore treated as **analytical groupings rather than official product categories**.

These four groups form the basis for **Task 3 — Generative AI Product-Category Summaries**, where the most reviewed products and their customer feedback are analyzed to generate category-level recommendation articles.

## Task 3 — Generative AI Category Summaries

The third task focused on using **Generative AI to transform customer review data into concise product-category recommendation articles**.

The objective was not simply to summarize individual reviews, but to combine customer feedback across multiple products and identify the main strengths, weaknesses, recurring complaints, and trade-offs within each product category.

### Data Preparation and Review Analysis

For each product category, the analysis was performed using the following information:

* product name
* review title
* review text
* review rating

As an intermediate step, the review titles and review texts were combined and converted into **TF-IDF features** using `TfidfVectorizer`.

For the **Tablets** category, for example, the analysis used:

* a maximum of **1,000 TF-IDF features**
* English stop-word removal
* unigrams and bigrams (`ngram_range=(1,2)`)

The TF-IDF representation was first used to identify the most relevant terms appearing across the reviews.

The reviews were then separated according to their star rating:

* **Positive reviews:** rating ≥ 4
* **Negative reviews:** rating ≤ 2

For Tablets, this resulted in:

* **16,105 positive reviews**
* **532 negative reviews**

The positive and negative TF-IDF scores were then compared. A term was considered particularly characteristic of one sentiment when its score was at least **1.5 times higher** than its score in the opposite sentiment group.

This allowed the analysis to identify recurring themes rather than relying only on individual reviews.

### Example: Tablet Review Themes

For the Tablets category, the analysis identified several recurring positive themes:

* **Value for money:** price, great price, value
* **Ease of use:** easy, easy use
* **Children:** kids, year old, son, daughter
* **Reading:** reading
* **Gifts:** gift, Christmas

The main negative themes included:

* **Performance:** slow
* **Battery and charging:** charge, charging, charger
* **Apps and software:** apps, app, download
* **Screen:** screen
* **Advertising:** ads
* **Connectivity:** Wi-Fi, internet
* **Reliability:** returned, disappointed

The same type of analysis was applied to the other product categories in order to provide the Generative AI model with structured information about recurring customer opinions.

### Experiments with Local Generative Models

Several pretrained text-generation and summarization models were tested before using an external LLM.

The purpose of these experiments was to determine whether the category-level articles could be generated locally using the available CPU-based hardware.

**FLAN-T5-small** was tested first. Its outputs were generally too short and generic to provide useful category-level recommendations.

**FLAN-T5-base** produced somewhat more meaningful results, but its quality decreased with longer inputs and it sometimes mixed positive and negative feedback.

**BART-large-CNN**, a model specifically designed for summarization, was also considered. However, its size made local installation and execution impractical on the available CPU-based hardware.

Finally, **DistilBART-CNN-6-6**, a smaller distilled BART model, was tested as a lighter alternative.

Although DistilBART was able to identify relevant information from the reviews, its outputs remained too close to individual source reviews and did not consistently synthesize information from multiple reviews into a coherent category-level recommendation.

For this reason, the local pretrained models were not used for the final generation stage.

### Final Generative AI Approach

After testing the local models, an external LLM approach was used with **Google Gemini**.

The Gemini API was successfully integrated into the project using the `google-genai` Python package.

For each product category, the **three products with the highest number of available reviews** were selected. Their average rating and customer-review information were then provided to Gemini.

The model was not asked simply to summarize each product independently. Instead, a structured prompt instructed it to produce a **comparative recommendation article**.

The prompt required the model to:

1. introduce the three products;
2. explain their main differences;
3. compare their strengths;
4. compare weaknesses and recurring customer complaints;
5. identify the best option for different types of buyers;
6. explain the main trade-offs;
7. provide a final recommendation.

The model was explicitly instructed to:

* use only the information supplied from the dataset;
* avoid inventing specifications or product features;
* avoid copying the review summaries verbatim;
* distinguish clearly between strengths and weaknesses;
* maintain an objective and balanced tone;
* compare the products directly rather than simply listing three independent summaries.

The same methodology was then adapted to each of the four product clusters identified in Task 2.

### Final Output

The final Generative AI stage produced **one recommendation article for each of the four product categories**:

* E-readers & Kindle
* Smart Home & Alexa
* Tablets
* Streaming & TV

The resulting articles combine quantitative information such as **review volume and average rating** with qualitative information extracted from customer feedback.

This approach allowed the project to move from raw customer reviews and unsupervised product clusters to a more human-readable form of **category-level product recommendation**.

### Notebook Cleanup

The notebook contains the complete development process, but some intermediate experiments and failed model attempts were removed from the final version once they had been evaluated.

This was done deliberately to keep the notebook readable and reproducible. Experiments that produced errors, unusable outputs, or approaches that were clearly abandoned were not kept as large blocks of non-functional code underneath the final solution.

The experimentation process is nevertheless documented in the analysis above, including the evaluation of different local summarization models and the final decision to use Gemini.

### Limitations

The quality of the final articles depends on the information available in the original dataset and on the quality of the review summaries provided to the generative model.

In particular, some category data is incomplete or contains highly aggregated product/category labels. Therefore, the generated articles should be interpreted as **data-driven recommendation summaries**, rather than independent professional product reviews.

The prompts were designed to minimize hallucinations by restricting Gemini to the information provided in the dataset and by explicitly prohibiting unsupported product specifications or features.

## Task 4 — Deployment

The fourth task focused on deploying the sentiment classification model as a **public web application**.

The final model selected in Task 1, based on **TF-IDF and Logistic Regression**, was saved so that it could be reused outside the Jupyter Notebook.

### Model Export

The trained model and TF-IDF vectorizer were exported using `joblib`:

```python
import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

These two files contain the trained components required by the application:

* `sentiment_model.pkl` — the trained Logistic Regression classifier
* `tfidf_vectorizer.pkl` — the TF-IDF vectorizer used to transform new reviews into the same numerical representation used during training

A test review was also used to verify that the saved model could classify new text and return class probabilities.

For example:

```python
test_review = "This tablet is excellent, easy to use and the battery lasts a long time."

X_test_review = vectorizer.transform([test_review])

prediction = model.predict(X_test_review)[0]

probabilities = model.predict_proba(X_test_review)[0]
```

The model successfully returned a sentiment prediction together with the probability associated with each class.

### Streamlit Web Application

The deployment stage was then moved outside the notebook.

A Python **Streamlit** application was created to provide a simple interface where users can enter an Amazon customer review and obtain a sentiment prediction.

The application:

1. loads the trained model;
2. loads the TF-IDF vectorizer;
3. accepts a customer review as text input;
4. transforms the review using the saved TF-IDF vectorizer;
5. predicts the sentiment;
6. displays the predicted class;
7. displays the confidence scores for the three sentiment classes.

The application was tested locally with Streamlit before deployment.

### GitHub Integration

The application source code and required model files were uploaded to a dedicated **GitHub repository**:

`nlp-customer-reviews`

The repository contains the Python application, the dependency file, and the serialized Machine Learning model and vectorizer required to run the application.

GitHub was then connected to **Streamlit Community Cloud**, which was used to deploy the application publicly.

### Public Application

The final application is publicly accessible at:

**https://nlp-customer-reviews-238iycgr2qn9qgg5quwzmw.streamlit.app/**

The deployed application allows users to enter their own customer review and receive an immediate sentiment classification without needing to run the Jupyter Notebook locally.

### Deployment Structure

The deployment process can be summarized as:

**Jupyter Notebook → trained model → `.pkl` files → Python/Streamlit application → GitHub → Streamlit Community Cloud → public web application**

This final step transformed the Machine Learning model developed during the project into an accessible application that can be tested directly through a web browser.
