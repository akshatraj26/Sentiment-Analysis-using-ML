---

# Sentiment Analysis Project

---

## Count Vectorizer Intuition

Count Vectorizer is a fundamental tool in Natural Language Processing (NLP) that transforms a collection of text documents into a matrix of token counts. Here's an intuitive explanation of how it works:

### What is Count Vectorizer?

Count Vectorizer converts text data into numerical data, which is necessary for machine learning algorithms to process textual information. The transformation is done by counting the occurrences of each word (or token) in a document and representing these counts as a matrix.

### How Does Count Vectorizer Work?

1. **Tokenization**: The text is split into individual words (tokens). For example, the sentence "I love machine learning" would be tokenized into ["I", "love", "machine", "learning"].

2. **Building the Vocabulary**: Count Vectorizer creates a dictionary of all unique tokens (words) across the entire corpus (collection of documents). This dictionary forms the vocabulary of the corpus.

3. **Counting the Tokens**: For each document, Count Vectorizer counts the number of occurrences of each word in the vocabulary. These counts are used to create a vector for each document.

4. **Creating the Document-Term Matrix**: Each document is represented as a vector where each element corresponds to the count of a particular word from the vocabulary. The entire corpus is thus transformed into a document-term matrix.

### Example

Consider a corpus with two documents:
1. "I love NLP"
2. "NLP is fun"

The vocabulary built from these documents would be:
["I", "love", "NLP", "is", "fun"]

The document-term matrix would look like this:

|         | I | love | NLP | is | fun |
|---------|---|------|-----|----|-----|
| Doc 1   | 1 | 1    | 1   | 0  | 0   |
| Doc 2   | 0 | 0    | 1   | 1  | 1   |

- "I love NLP" is represented as [1, 1, 1, 0, 0]
- "NLP is fun" is represented as [0, 0, 1, 1, 1]

### Why Use Count Vectorizer?

- **Simplicity**: It's easy to understand and implement.
- **Baseline Method**: Often used as a starting point before moving to more complex methods like TF-IDF or word embeddings.
- **Sparse Representations**: Efficiently handles large vocabularies and sparse data.

Count Vectorizer is a powerful tool for converting text into a format suitable for machine learning algorithms, providing a foundation for further text analysis and modeling.

---


---

## TF-IDF Intuition

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It is widely used in information retrieval and text mining to reflect how important a word is to a document in a corpus.

### What is TF-IDF?

TF-IDF is a numerical statistic that is intended to reflect the importance of a word in a document relative to the entire corpus. It balances the frequency of a word in a document with its rarity across the entire corpus, helping to highlight words that are more informative and less common.

### How Does TF-IDF Work?

TF-IDF is a product of two statistics: Term Frequency (TF) and Inverse Document Frequency (IDF).

1. **Term Frequency (TF)**:
    - Measures how frequently a word appears in a document.
    - Calculated as the number of times a word appears in a document divided by the total number of words in that document.
    - Formula: \( \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} \)

2. **Inverse Document Frequency (IDF)**:
    - Measures how important a word is in the entire corpus.
    - Calculated as the logarithm of the total number of documents divided by the number of documents containing the word.
    - Formula: \( \text{IDF}(t) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents containing term } t} \right) \)

3. **TF-IDF Calculation**:
    - The TF-IDF score for a word in a document is the product of its TF and IDF scores.
    - Formula: \( \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) \)

### Example

Consider a corpus with three documents:
1. "I love machine learning"
2. "Machine learning is amazing"
3. "I love amazing machines"

Let's calculate the TF-IDF for the word "machine" in the first document.

1. **Term Frequency (TF)** for "machine" in Document 1:
    - TF(machine, Doc 1) = 1/4 = 0.25

2. **Inverse Document Frequency (IDF)** for "machine":
    - "machine" appears in 2 out of 3 documents.
    - IDF(machine) = \( \log(3/2) = 0.176 \)

3. **TF-IDF** for "machine" in Document 1:
    - TF-IDF(machine, Doc 1) = 0.25 * 0.176 = 0.044

### Why Use TF-IDF?

- **Relevance**: TF-IDF helps to identify words that are important and relevant to a document, reducing the impact of commonly used words (stop words).
- **Weighting**: It assigns a higher weight to words that are frequent in a specific document but rare in the corpus, enhancing the distinction between documents.
- **Information Retrieval**: TF-IDF is commonly used in search engines and information retrieval systems to rank documents based on their relevance to a query.

TF-IDF is a powerful tool for transforming text data into meaningful numerical representations, enabling effective information retrieval and text analysis.

---


## Setting Up the Environment

1. Create a virtual environment:
    ```bash
    python -m venv .env
    ```

2. Activate the virtual environment:
    - On Windows:
        ```bash
        .env\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source .env/bin/activate
        ```

## Installing Dependencies

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

