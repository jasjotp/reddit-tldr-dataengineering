import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import boto3
import io 
import sys
import os
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region

def get_combined_data():
    '''
    connects to Amazon S3, grabs all of the files in the S3 bucket and combines them, before cleaning the data and creating nccessessary columns for EDA
    '''
    # create the session 
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    # initialize the s3 client
    s3 = session.client('s3')

    # prefix for S3 folder to pull from 
    prefix = 'raw/'

    response = s3.list_objects_v2(Bucket=aws_bucket_name, Prefix=prefix)

    files = [f['Key'] for f in response.get('Contents', []) if f['Key'].endswith('.csv')]

    # Load and combine
    dfs = []

    for file_key in files:
        obj = s3.get_object(Bucket=aws_bucket_name, Key=file_key)

        df = pd.read_csv(io.BytesIO(obj['Body'].read()))

        dfs.append(df)

    # Combine into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(dfs)} files. Combined shape: {combined_df.shape}")
    print(combined_df.head())

    # load the combined df with all recent day's data into a csv for use in different files 
    combined_df.to_csv('reddit_combined_data.csv')

    # check all posts that are null or blank (have spaces or just newlines), and clean and prepare all columns needed for EDA
    nulls = combined_df['selftext'].isnull()
    blanks = combined_df['selftext'].fillna('').str.strip() == ''
    removed_or_deleted = combined_df['selftext'].isin(['[removed]', '[deleted]']) # filter for posts that contain the reddit removed and deleted tags
    empty_posts = nulls | blanks | removed_or_deleted
    combined_df = combined_df.rename(columns = {'selftext': 'body'})
    combined_df['created_dt'] = pd.to_datetime(combined_df['created_utc'])
    combined_df['hour'] = combined_df['created_dt'].dt.hour
    combined_df['weekday'] = combined_df['created_dt'].dt.day_name()
    combined_df['word_count'] = combined_df['body'].str.split().str.len()
    combined_df['body'] = combined_df['body'].fillna("") # fill all blank bodies with empty strings so the vectorizer can successfully analyze them

    return combined_df

# EDA 
if __name__ == "__main__":
    combined_df = get_combined_data()
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
    ax = sns.countplot(x = 'hour', hue='hour', data = combined_df, order = sorted_hourly_counts, palette = 'Blues_d', legend=False)

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=10, fontweight='bold')

    # add the title and labels to the box plot 
    plt.title('Reddit Post Frequency by Hour of the Day', fontsize = 16)
    plt.xlabel('Hour (0-23)', fontsize = 12)
    plt.ylabel('Number of Posts', fontsize = 12)
    plt.xticks(ticks=range(len(sorted_hourly_counts)), labels=[hour_labels[h] for h in sorted_hourly_counts])
    plt.tight_layout()
    plt.savefig('graphs/post_frequency_by_hour.png', bbox_inches='tight')
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
    plt.savefig('graphs/avg_score_comments_by_hour.png', bbox_inches='tight')
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
    plt.savefig('graphs/avg_score_comments_by_wordcount.png', bbox_inches='tight')
    plt.close()

    ### It looks like posts with around < 50 words have been getting the most engagement (highest scores and number of comments)

    ### 4. Does posting on a specific day of the week affect post score?

    sns.boxplot(data=combined_df, x='weekday', y='score', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    plt.title('Scores by Day of the Week', fontsize=16)
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('graphs/avg_score_by_weekday.png', bbox_inches='tight')
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
    plt.savefig('graphs/top30_common_words_all_posts.png', bbox_inches='tight')

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
    plt.savefig('graphs/highest_post_top30_tf-idf_words.png', bbox_inches='tight')

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
    plt.savefig('graphs/lowest_post_top30_tf-idf_words.png', bbox_inches='tight')

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
    plt.savefig('graphs/tfidf_high_vs_low_difference_comparison.png', bbox_inches='tight')

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
    plt.savefig('graphs/tfidf_high_vs_low_ratio_comparison.png', bbox_inches='tight')

    ### 8. Check correlations between numeric features
    plt.figure(figsize=(14, 8))
    sns.heatmap(combined_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig("graphs/correlation_matrix.png", bbox_inches='tight')
    plt.close()