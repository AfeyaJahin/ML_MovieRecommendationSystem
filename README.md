**README: Enhancing Movie Discovery - A Comparative Analysis of Recommendation Systems**

## Project Overview
This project presents a comprehensive analysis of three different recommendation system approaches for movies: Content-Based Filtering, Hybrid Collaborative Filtering, and K-Nearest Neighbors (KNN). Each method aims to improve movie discovery and user engagement by predicting and recommending personalized titles.

**Models Implemented:**
1. **Content-Based Filtering:**  
   Uses cosine similarity and weighted averages to suggest movies that share similarities with the ones a user has previously enjoyed.
   
2. **Hybrid Collaborative Filtering:**  
   Combines deep learning neural networks with matrix factorization, using user-item interactions and metadata to recommend personalized titles.

3. **K-Nearest Neighbors:**  
   Classifies unseen movies as either "popular" or "not popular," based on a binary popularity score threshold, relying on input features like budget, revenue, runtime, etc.

## Dataset
The dataset used for this project is "The Movies Dataset" from Kaggle:
- **movies_metadata.csv:** Information on over 45,000 films.
- **keywords.csv:** Plot keywords for movies in JSON format.
- **credits.csv:** Cast and crew details.
- **ratings_small.csv:** A subset of ratings by 700 users on 9,000 movies.

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook (optional, but useful for exploration)

### Dependencies
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- TensorFlow/Keras
- scikit-learn
- pandas
- numpy

### Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Enhancing-Movie-Discovery.git
   ```
2. **Explore Models:**
   - **Content-Based Filtering:**
     - Check the Jupyter Notebook or Python file with the content-based model implementation.
     - Adjust parameters like the TF-IDF vectorization and weighted average scoring.

   - **Hybrid Collaborative Filtering:**
     - Review the neural network-based hybrid model in the respective notebook/Python file.
     - Tune hyperparameters and explore the TensorFlow Recommenders (TFRS) API.

   - **K-Nearest Neighbors:**
     - Check out the KNN implementation to classify movies by popularity.
     - Experiment with different K values and analyze the classification report.

3. **Evaluation:**
   - Refer to the evaluation results provided or run the models to see their performance using metrics like RMSE and accuracy.


## References
1. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) on Kaggle
2. Related research papers (see the final report for full references)

