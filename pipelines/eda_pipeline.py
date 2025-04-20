import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px
import boto3
import io
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer

# set the current and output dirs for Airflow and make the output directory if it does not exist already 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../data/output")
GRAPHS_DIR = os.path.join(CURRENT_DIR, "../graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region
from utils.s3_helpers import upload_graph_to_s3, load_combined_data_from_s3

def run_reddit_eda():

    # read the updated combined data using the load_combined_data heper function from utils to get the updated (daily) data
    combined_df = load_combined_data_from_s3(aws_access_key, aws_secret_access_key, aws_region, aws_bucket_name)

    ### 1. Find what times users have been posting the most for most recent days 

    # create a 24-hour → 12-hour label map
    hour_labels = {
        0: '12 AM', 1: '1 AM', 2: '2 AM', 3: '3 AM', 4: '4 AM', 5: '5 AM',
        6: '6 AM', 7: '7 AM', 8: '8 AM', 9: '9 AM', 10: '10 AM', 11: '11 AM',
        12: '12 PM', 13: '1 PM', 14: '2 PM', 15: '3 PM', 16: '4 PM',
        17: '5 PM', 18: '6 PM', 19: '7 PM', 20: '8 PM', 21: '9 PM',
        22: '10 PM', 23: '11 PM'
    }

    sorted_hourly_counts = combined_df['hour'].value_counts().sort_values(ascending=False).index

    # plot the post frequency per hour of the day 
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x = 'hour', hue='hour', data = combined_df, order = sorted_hourly_counts, palette = 'Blues_d')
    ax.legend().set_visible(False)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=10, fontweight='bold')

    # add the title and labels to the box plot 
    plt.title('Reddit Post Frequency by Hour of the Day', fontsize = 16)
    plt.xlabel('Hour (0-23)', fontsize = 12)
    plt.ylabel('Number of Posts', fontsize = 12)
    plt.xticks(ticks=range(len(sorted_hourly_counts)), labels=[hour_labels[h] for h in sorted_hourly_counts])
    plt.tight_layout()
   
    post_frequency_by_hour_path = os.path.join(GRAPHS_DIR, 'post_frequency_by_hour.png')
    plt.savefig(post_frequency_by_hour_path, bbox_inches='tight')
    upload_graph_to_s3(post_frequency_by_hour_path)
    plt.close()

    ### The most posts are being posted at 5PM recently (UTC time)

    ### 2. Find what time of day gets the highest engagement (times when  the average score and comments are the highgest)
    # calculate average score and comments by hour
    hourly_avg_score_counts = combined_df.groupby('hour')[['score', 'num_comments']].mean().reset_index()

    print(f'Hourly Avg: \n{hourly_avg_score_counts}')

    # melt the dataframe so that we can draw a linegraph with a line for average score and num_comments per hour 
    hourly_melted = hourly_avg_score_counts.melt(id_vars='hour', value_vars=['score', 'num_comments'], var_name='Metric', value_name='Average')

    # plot the lines for the average score and number of comments per hour 
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly_melted, x='hour', y='Average', hue='Metric', marker='o')

    plt.title('Average Score & Comments by Hour', fontsize=16)
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average', fontsize=12)
    plt.xticks(ticks=range(len(hourly_avg_score_counts)), labels=[hour_labels[h] for h in hourly_avg_score_counts.index])
    plt.tight_layout()
       
    avg_score_comments_by_hour_path = os.path.join(GRAPHS_DIR, 'avg_score_comments_by_hour.png')
    plt.savefig(avg_score_comments_by_hour_path)
    upload_graph_to_s3(avg_score_comments_by_hour_path)
    plt.close()

    ### Posts see the most engagement around 5AM UTC Time

    # show the above average scores per hour in plotly so we have tool tips and can see precise values for each hour
    fig = px.line(hourly_melted, x='hour', y='Average', color='Metric', markers=True)
    fig.write_html("graphs/avg_score_comments_by_hour.html")

    ### 3. See if longer posts are more engaging

    wordcount_score_commmets = (combined_df
                            .groupby('word_count')[['score', 'num_comments']]
                            .mean().reset_index()
                            .sort_values(by=['score', 'num_comments'], ascending=False)
                            .head(20))

    print(f"Average scores/comments per Post Wordcount: \n{wordcount_score_commmets}")

    # melt the dataframe so that we can draw a linegraph with a line for average score and num_comments per hour 
    wordcount_melted = wordcount_score_commmets.melt(id_vars='word_count', value_vars=['score', 'num_comments'], var_name='Metric', value_name='Average')

    # plot the lines for the average score and number of comments per hour 
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=wordcount_melted, x='word_count', y='Average', hue='Metric', marker='o')

    plt.title('Average Score & Comments by Post Wordcount', fontsize=16)
    plt.xlabel('Wordcount of Post', fontsize=12)
    plt.ylabel('Average', fontsize=12)
    plt.tight_layout()

    avg_score_comments_by_wordcount_path = os.path.join(GRAPHS_DIR, 'avg_score_comments_by_wordcount.png')
    plt.savefig(avg_score_comments_by_wordcount_path)
    upload_graph_to_s3(avg_score_comments_by_wordcount_path)
    plt.close()

    ### It looks like posts with around < 50 words have been getting the most engagement (highest scores and number of comments)

    ### 4. Does posting on a specific day of the week affect post score?

    sns.boxplot(data=combined_df, x='weekday', y='score', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    plt.title('Scores by Day of the Week', fontsize=16)
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.tight_layout()

    avg_score_by_weekday_path = os.path.join(GRAPHS_DIR, 'avg_score_by_weekday.png')
    plt.savefig(avg_score_by_weekday_path)
    upload_graph_to_s3(avg_score_by_weekday_path)
    plt.close()

    ### Based on the engagement data from r/dataengineering, posts made on Saturdays consistently outperform those made on weekdays — both in terms of average score and likelihood of going viral (score > 100)

    ### 5. Find the 30 most common words in all Reddit posts

    # custom stopwords to remove in addition to English ones that do not give us any insight
    custom_stopwords = [
        'like', 'use', 'just', 've', 'using', 'want', 'dom', 'as well'
    ]

    # combine default english stopwrods with the custom stop words 
    all_stopwords = list(text.ENGLISH_STOP_WORDS.union(custom_stopwords))

    # use CountVectorizer - initialize CountVectorizer and limit it to the top 30 features/words
    vectorizer = CountVectorizer(stop_words=all_stopwords, max_features=30)

    # ensure again that there are no null values beofre vectorzing 
    combined_df['body'] = combined_df['body'].fillna('').astype(str)

    # transform the body columns 
    X = vectorizer.fit_transform(combined_df['body']) # learn vocabulary of bodies of posts

    # sum the word occurrences 
    word_freq_total = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    word_counts_total = word_freq_total.sum().sort_values(ascending=False)

    # plot the bar chart to see the mostfrequent words in each post 
    plt.figure(figsize=(14, 8))
    word_counts_total.plot(kind='bar', color='orangered')

    plt.title("Top 30 Most Common Words in All Reddit Posts in r/dataengineering Recently", fontsize=16)
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    top30_common_words_path = os.path.join(GRAPHS_DIR, 'top30_common_words_all_posts.png')
    plt.savefig(top30_common_words_path)
    upload_graph_to_s3(top30_common_words_path)
    plt.close()

    ### 6. use TfidfVectorizer to find words that commonly appear in high engagement posts but do not appear much in other lower engagement posts
    # find the top 15% of highest engagement posts 
    high_engagement_posts = combined_df[combined_df['score'] > combined_df['score'].quantile(0.85)]

    tfidf_vectorizer = TfidfVectorizer(stop_words=all_stopwords, max_features=30)
    X_high = tfidf_vectorizer.fit_transform(high_engagement_posts['body']) # learn vocabulary of top 15% posts - returns 2D matrix of feature names and and each post's TF IDF score for each word (importance per word)

    top_post_word_score = pd.DataFrame(X_high.toarray(), columns=tfidf_vectorizer.get_feature_names_out()) # to get feature/column names 

    top_post_word_scores = top_post_word_score.sum().sort_values(ascending=False)

    # plot the bar chart to see the most important words in high engagement (top 15%) posts
    plt.figure(figsize=(14, 8))
    top_post_word_scores.plot(kind='bar', color='orangered')

    plt.title("Top 30 TF-IDF Words in Top 15 Percent of Reddit Posts in r/dataengineering Recently", fontsize=16)
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    top30_tf_idf_words_path = os.path.join(GRAPHS_DIR, 'highest_post_top30_tf-idf_words.png')
    plt.savefig(top30_tf_idf_words_path)
    upload_graph_to_s3(top30_tf_idf_words_path)
    plt.close()
    
    ### 7. use TfidfVectorizer to find words that commonly appear in low engagement posts but do not appear much in other highlower engagement posts
    # find the lowest 15% of posts by engagement (scores less than the 15th percentile of scores)
    low_engagement_posts = combined_df[combined_df['score'] < combined_df['score'].quantile(0.30)]

    tfidf_vectorizer_low = TfidfVectorizer(stop_words=all_stopwords, max_features=30)

    X_low = tfidf_vectorizer_low.fit_transform(low_engagement_posts['body']) # learn vocabulary of top 15% posts - returns 2D matrix of feature names and and each post's TF IDF score for each word (importance per word)

    lowest_post_word_score = pd.DataFrame(X_low.toarray(), columns=tfidf_vectorizer_low.get_feature_names_out()) # to get feature/column names 

    # sum all the scores of the lowest engagement post words and sore in descending order
    low_post_word_scores = lowest_post_word_score.sum().sort_values(ascending=False)

    # plot the bar chart to see the most important words in high engagement (top 15%) posts
    plt.figure(figsize=(14, 8))
    low_post_word_scores.plot(kind='bar', color='orangered')

    plt.title("Top 30 TF-IDF Words in Lowest 30 Percent of Reddit Posts in r/dataengineering Recently", fontsize=16)
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    lowest_post_top30_tf_idf_words_path = os.path.join(GRAPHS_DIR, 'lowest_post_top30_tf-idf_words.png')
    plt.savefig(lowest_post_top30_tf_idf_words_path)
    upload_graph_to_s3(lowest_post_top30_tf_idf_words_path)
    plt.close()

    # create a combined df where each word is scored for its high engagement vs low engagement posts
    tfidf_comparison = pd.DataFrame({
        'High Engagement': top_post_word_scores,
        'Low Engagement': low_post_word_scores
    }).fillna(0)  # for missing words, full them with 0 

    # normalize TF-IDF scores to account for group size difference
    tfidf_comparison['High Engagement'] /= len(high_engagement_posts)
    tfidf_comparison['Low Engagement'] /= len(low_engagement_posts)

    # calculate the differennce for each word or how much the word appears more in high vs low engagmeent posts
    tfidf_comparison['Difference'] = tfidf_comparison['High Engagement'] - tfidf_comparison['Low Engagement']

    # sort the words by the difference in descending order
    top_unique_words = tfidf_comparison.sort_values(by='Difference', ascending=False).head(15)

    # plot the differences for each high engagement and low engagement word 
    top_unique_words[['High Engagement', 'Low Engagement']].plot(
        kind='bar',
        figsize=(14, 7),
        color=['green', 'red']
    )

    plt.title("TF-IDF Difference Comparison of Top Words: High vs Low Engagement Posts")
    plt.ylabel("Total TF-IDF Score")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Word")
    plt.tight_layout()

    tfidf_high_vs_low_difference_path = os.path.join(GRAPHS_DIR, 'tfidf_high_vs_low_difference_comparison.png')
    plt.savefig(tfidf_high_vs_low_difference_path)
    upload_graph_to_s3(tfidf_high_vs_low_difference_path)
    plt.close() 

    # calculate the ratio for each word that shows up in high engagement posts vs low engageemnt posts
    tfidf_comparison['Ratio'] = tfidf_comparison['High Engagement'] / (tfidf_comparison['Low Engagement'] + 1e-6)

    # get the natural log of the ratio to normalize values for visualization 
    tfidf_comparison['Log Ratio'] = np.log1p(tfidf_comparison['Ratio']) 

    # sort the top 15 ratio words in descending order
    top_ratio_words = tfidf_comparison.sort_values(by='Log Ratio', ascending=False).head(15)

    # plot the ratios for each high engagement and low engagement word 
    top_ratio_words['Log Ratio'].plot(
        kind='bar',
        figsize=(14, 7),
        color=['red'],
    )

    plt.title("TF-IDF Logged Ratio Comparison of Top Words: High vs Low Engagement Posts")
    plt.ylabel("TF-IDF Logged Ratio")
    plt.xticks(rotation=45, ha='right')
    plt.legend().set_visible(False)
    plt.xlabel("Word")
    plt.tight_layout()

    tfidf_high_vs_low_ratio_comparison_path = os.path.join(GRAPHS_DIR, 'tfidf_high_vs_low_ratio_comparison.png')
    plt.savefig(tfidf_high_vs_low_ratio_comparison_path)
    upload_graph_to_s3(tfidf_high_vs_low_ratio_comparison_path)
    plt.close()  

    ### 8. Check correlations between numeric features
    plt.figure(figsize=(14, 8))
    sns.heatmap(combined_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()

    correlation_matrix_path = os.path.join(GRAPHS_DIR, 'correlation_matrix.png')
    plt.savefig(correlation_matrix_path)
    upload_graph_to_s3(correlation_matrix_path)
    plt.close()  