# TL;DR: What's Trending on r/dataengineering?

This project analyzes Reddit posts from the `r/dataengineering` subreddit to identify trending discussion topics, using daily ingested data and a custom-built NLP clustering and sentiment pipeline.

The goal is to help engineers and curious readers quickly identify what topics are receiving high engagement and understand what makes those posts stand out.

## Overview

This project uses Apache Airflow to orchestrate a daily ETL pipeline that:

1. Ingests Reddit posts via the Reddit API or Pushshift.
2. Stores the data in Amazon S3 and SQLite/CSV.
3. Applies an NLP pipeline to:
   - Clean and tokenize post content.
   - Vectorize using Word2Vec and TF-IDF.
   - Cluster using KMeans with elbow detection.
   - Visualize keyword clusters.
   - Assign sentiment using Hugging Face Transformers.
   - Generate cluster-level and post-level insights.

All outputs (plots, cluster summaries, sentiment insights) are saved locally and also uploaded to Amazon S3.

## Pipeline Features

- Automatically downloads and processes Reddit post data daily using Apache Airflow.
- Uses Word2Vec + KMeans to cluster similar word themes.
- Measures how each cluster correlates with post score, comment count, readability, and engagement.
- Labels each post as high/mid/low engagement based on score quantiles.
- Visualizes clusters and heatmaps for interpretability.
- Assigns each post to a dominant cluster based on keyword matches.

## Tech Stack

### ETL & Orchestration
- Apache Airflow
- Amazon S3

### Data Storage
- CSV

### NLP & Machine Learning
- Python (pandas, NumPy, scikit-learn, re)
- Gensim (Word2Vec)
- Hugging Face Transformers (DistilBERT)
- TF-IDF (scikit-learn)
- KMeans Clustering with Elbow Detection (`kneed`)
- NLTK (stopwords)

### Visualization
- Matplotlib
- Seaborn
- Plotly
- Scipy (ConvexHull)

## Outputs

- `cluster_engagement_insights.csv`: Summary of each word cluster and its average metrics.
- `post_engagement_insights.csv`: Each post enriched with engagement level, dominant topic cluster, and metrics.
- `word2vec_clusters.png`: Clustered keywords plotted in 2D space.
- `cluster_engagement_metrics.png`: Heatmap of post metrics across clusters.
- `top_10_words_by_cluster.png`: TF-IDF-weighted top keywords per cluster.

## How to Run

1. Clone the repository and set up your environment.
2. Add your Reddit API credentials.
3. Configure Airflow and S3 credentials inside `/utils/constants.py`.
4. Trigger the DAG via Airflow UI or run `run_nlp_clustering_pipeline()` manually.

