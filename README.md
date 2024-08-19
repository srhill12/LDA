# BBC News Articles Topic Modeling with LDA

This project demonstrates the application of Latent Dirichlet Allocation (LDA) to perform topic modeling on a dataset of BBC news articles. The objective is to identify distinct topics within the articles and assign appropriate labels to them based on the most frequent words associated with each topic.

## Project Overview

### Objectives

1. **Data Preprocessing**:
   - Clean and preprocess the news articles by removing numbers and non-alphabetic characters.
   - Apply tokenization and vectorization using `CountVectorizer`.

2. **Topic Modeling**:
   - Use Latent Dirichlet Allocation (LDA) to discover hidden topics within the news articles.
   - Identify the top 15 words for each topic and assign relevant labels to them.

3. **Topic Assignment**:
   - Assign the most relevant topic to each news article based on the LDA results.
   - Add these topics and their corresponding labels to the dataset.

## Tools and Libraries

- **Python**: Programming language used for the entire project.
- **Pandas**: Library used for data manipulation and analysis.
- **NumPy**: Library used for numerical computations.
- **Regular Expressions (re)**: Used for data cleaning and preprocessing.
- **Scikit-learn**: Machine learning library used for vectorization and topic modeling.

## Key Steps and Functions

### 1. Data Preprocessing

- **Loading Data**:
  - Load the dataset containing BBC news articles from a CSV file using Pandas.
  - Example:
    ```python
    news_articles_df = pd.read_csv('Resources/bbc_news_articles.csv')
    ```

- **Cleaning Data**:
  - Remove numbers and non-alphabetic characters from the `news_summary` column using regular expressions.
  - Example:
    ```python
    news_articles_df['news_summary'] = news_articles_df['news_summary'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
    ```

### 2. Vectorization and Topic Modeling

- **Vectorization**:
  - Use `CountVectorizer` to convert the text data into a Document-Term Matrix (DTM) with specific parameters (`max_df=0.95`, `min_df=5`, and `stop_words='english'`).
  - Example:
    ```python
    cv = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    dtm = cv.fit_transform(news_articles_df['news_summary'])
    ```

- **Latent Dirichlet Allocation (LDA)**:
  - Apply LDA with 5 topics to identify different themes within the news articles.
  - Example:
    ```python
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA_data = LDA.fit(dtm)
    ```

- **Top Words for Each Topic**:
  - Extract and print the top 15 words associated with each of the 5 topics identified by LDA.
  - Example Output:
    ```python
    The Top 15 Words For Topic #1: ['actress', 'years', 'told', 'time', 'star', ...]
    ```

### 3. Topic Assignment

- **Topic Transformation**:
  - Transform the DTM into topic probabilities for each document to determine the most likely topic for each article.
  - Example:
    ```python
    topic_results = LDA.transform(dtm)
    ```

- **Add Topic Labels**:
  - Assign topics and labels to each article by finding the topic with the highest probability and mapping it to a label.
  - Example Function:
    ```python
    def add_topic_labels(df, topic_results, topic_labels):
        df['topic'] = topic_results.argmax(axis=1) + 1
        df['topic_label'] = df['topic'].map(topic_labels)
    ```

- **Topic Labels**:
  - The topics have been labeled as follows:
    - Topic 1: Entertainment
    - Topic 2: Sports
    - Topic 3: Business
    - Topic 4: Politics
    - Topic 5: Technology

### 4. Final Output

- The DataFrame is updated with the new columns for topics and their labels. The first and last 20 rows are displayed to validate that the articles have been appropriately categorized.

## Results

- **Topic Categorization**:
  - The LDA model successfully identified and categorized the news articles into distinct topics based on their content.
  - The assigned labels accurately reflect the main themes of the articles, as observed from the top words and the topics.

## Requirements

- Python 3.x
- Libraries: Pandas, NumPy, scikit-learn, re

## Installation

To install the required libraries, use the following pip command:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

1. Ensure all required libraries are installed.
2. Load the dataset and preprocess the text data.
3. Apply `CountVectorizer` to create a Document-Term Matrix.
4. Use Latent Dirichlet Allocation (LDA) to identify topics.
5. Assign topics and labels to each article and display the results.

## Conclusion

This project demonstrates the effectiveness of using LDA for topic modeling on textual data. The model was able to identify distinct topics within the BBC news articles, and the results were appropriately labeled, providing a clear understanding of the underlying themes in the dataset.
