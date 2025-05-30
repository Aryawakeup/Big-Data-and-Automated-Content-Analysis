# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:44:08 2025

@author: Lenovo
"""

#%%

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import lyricsgenius
import numpy as np
import math
#%%
#Get the dataset from Kaggle
top_songs_path = kagglehub.dataset_download("muhmores/spotify-top-100-songs-of-20152019")

print("Path to dataset1 files:", top_songs_path)


csv_top_songs_path = os.path.join(top_songs_path, "Spotify 2010 - 2019 Top 100.csv")

df_top_songs = pd.read_csv(csv_top_songs_path, encoding='utf-8-sig')

#Check the dataset
print(df_top_songs.head())
print(df_top_songs.columns)

#Get the brief information of the dataset, to better know how to analyze
print("Top Songs Dataset:")
print(df_top_songs.info())
print(df_top_songs[["title", "artist"]].isnull().sum())
print(df_top_songs[["title", "artist"]].dtypes)

df_top_songs = df_top_songs.dropna(subset=["title", "artist"])
#%% Optional, for the second time when the dataset is already cleaned and done, and no need to connect API again
#df_full = pd.read_csv ("cleaned_result.csv")
#df_lyrics =pd.read_csv ("lyrics_result.csv")
#%%
#Prepare for the API
songs = df_top_songs[["title", "artist"]].to_dict("records")

genius = lyricsgenius.Genius("kwYty6YeEgfbxUE8HtnAD0b_gdTiw6Lvm4jMY1Q_s5w6gsw-2mlBiAgg3-bE0kHl")
genius.skip_non_songs = True
genius.remove_section_headers = True

#%%
#Set a empty list, make the counter start from 0
all_lyrics = []
i = 0

#Make a loop to extract the name of artist and song, searching for matching lyrics
while i < len(songs):
    song = songs[i]
    title = str(song["title"])
    artist = str(song["artist"])

#To observe which the process of the lyrics scraping
    print(f"[{i+1}/{len(songs)}] got：{title} by {artist}")

    try:
        result = genius.search_song(title=title, artist=artist)
        lyrics = result.lyrics if result else None
    except Exception as e:
        print(f"wrong：{e}") #To make sure I didn't miss anything and I can retry again
        lyrics = None

    all_lyrics.append({
        "title": title,
        "artist": artist,
        "lyrics": lyrics
    })

    i += 1

#Save the data for the second time open the file cause this takes me 2.5 hours to got all the lyrics and I definitely don't want one more time
df_lyrics = pd.DataFrame(all_lyrics)
df_lyrics.to_csv("lyrics_result.csv", index=False)

print("Saved to lyrics_result.csv")

#%%
#To check the matching rate
matched = df_lyrics['lyrics'].notna().sum()
total = len(df_top_songs)
match_rate = matched / total * 100

print("Matched:", matched)
print("Total:", total)
print("Matching Rate: {:.2f}%".format(match_rate))
    
#%%
#To better help with the lyrics scraping, I tried to standardize song titles and remove noise information 
def clean_title(title):
    if pd.isna(title):
        return "" #If title is NaN, return an empty string directly to avoid error
    title = title.lower()
    title = re.sub(r"\(.*?version.*?\)", "", title) #Remove various bracket version marks one by one
    title = re.sub(r"\(.*?remix.*?\)", "", title)
    title = re.sub(r"\(.*?live.*?\)", "", title)
    title = re.sub(r"\(.*?edit.*?\)", "", title)
    title = re.sub(r"\(.*?\)", "", title)
    #Some of theses titles are more annoying!
    #These few lines of code are used to deal with the absence of brackets, but connected with -
    title = re.sub(r"- .*version", "", title) 
    title = re.sub(r"- live", "", title)
    title = re.sub(r"- remix", "", title)
    title = re.sub(r"- acoustic", "", title)
    title = re.sub(r"- .*", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    #Merge multiple spaces into one space, remove leading and trailing spaces, and capitalize the first letter of each word
    return title.title()

#Check how many songs are left now
missing = df_lyrics[df_lyrics["lyrics"].isna() | (df_lyrics["lyrics"].str.strip() == "")]

#Extract songs with missing lyrics
for i, row in missing.iterrows():
    raw_title = row["title"]
    cleaned_title = clean_title(raw_title)
    artist = row["artist"]
 
#Special Treat
    if cleaned_title.lower().strip() == "hot n*gga":
            cleaned_title = "Hot Nigga"
      
#Speical Treat 2
    if cleaned_title.lower().strip() == "if you're over me" and artist.lower() in ["years & years", "years and years"]:
        print("Manually search for：If You're Over Me")
        new_lyrics = genius.lyrics(song_url="https://genius.com/Years-and-years-if-youre-over-me-lyrics")
#I spent a long time investigating why I couldn't find the lyrics for the song If You're Over Me. 
#Finally, I asked gpt and found out that it was because the encoding of the fancy quotation mark (U+2019) didn't match.
    else:
        print(f"Try again {i}: {cleaned_title} - {artist}")
        try:
            result = genius.search_song(title=cleaned_title, artist=artist)
            new_lyrics = result.lyrics if result else None
        except Exception as e:
            print(f"Error: {e}")
            new_lyrics = None

    df_lyrics.at[i, "lyrics"] = new_lyrics
    
#Finally I got all the lyrics!
df_lyrics.to_csv("lyrics_result.csv", index=False)
print("Saved to lyrics_result.csv")
#%%
#...But, this not the end of dataset cleaning, there are more rows than I expected --duplicated lyrics!
#I originally wanted to merge the two tables directly, but there was a problem with duplicate lyrics, so I decided to create a new table and then use index matching
#Before that, check whether the indexes of the two tables can be aligned
df_top_songs.duplicated(subset=["title", "artist", "top year"], keep=False).sum()
df_top_songs[df_top_songs.duplicated(subset=["title", "artist", "top year"], keep=False)].sort_values(["title", "artist"])

#Check one more time every song has its lyric
matched = df_lyrics['lyrics'].notna().sum()
total = len(df_top_songs)
match_rate = matched / total * 100

print("Matched:", matched)
print("Total:", total)
print("Matching Rate: {:.2f}%".format(match_rate))

#Make sure the title in the two tables correspond row by row
(df_top_songs["title"] == df_lyrics["title"]).value_counts()

#%%
#To avoid the duplicated issue, just combine two dataset by index
df_top_songs["song_id"] = df_top_songs.index
df_lyrics["song_id"] = df_lyrics.index
df_full = df_top_songs.merge(df_lyrics, on="song_id", how="left")
print(df_full.columns)
(df_full["title_x"] == df_full["title_y"]).value_counts()

#%%
#Clean the dataset and rename the columns
df_full = df_full.drop(columns=['lyrics_len','title_y', 'artist_y','title_clean', 'artist_clean'])
df_full = df_full.rename(columns={"artist_x": "artist"})
df_full = df_full.rename(columns={"title_x": "title"})

#Finally! This is the FINAL CLEAN DATASET YAYYYY
df_full.to_csv("cleaned_result.csv", index=False)
print("Saved to cleaned_result.csv")
#%%
#Double check there's nothing missing cause I am insecure
print(df_full.isna().sum())

print(df_full.info())
print(df_top_songs.head())
df_full['year released'] = df_full['year released'].astype(int)
df_full['top year'] = df_full['top year'].astype(int) #Put the year into a usable, reasonable format without decimal points
#%%
#Exploratory analysis
#Initially check the trend and distribution of the music
#How many different genres are there each year
genre_diversity = df_full.groupby('top year')['top genre'].nunique()

plt.figure(figsize=(10, 5))
genre_diversity.plot(kind='bar', color='skyblue')
plt.title("Genre Diversity per Year")
plt.ylabel("Number of Unique Genres")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

#Genre analysis
top_genres = df_full['top genre'].value_counts().nlargest(10)
print(top_genres)
#Merge the remaining genres into "others"
others = df_full['top genre'].value_counts().iloc[15:].sum()
top_genres['Others'] = others

#Make the pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Genres Distribution')
plt.axis('equal') #Make it a cute round
plt.tight_layout() #Avoid overlap
plt.show()

#Visualize genre trend
# Top genres
common_genres = df_full['top genre'].value_counts().head(7).index
filtered = df_full[df_full['top genre'].isin(common_genres)]

# Countplot of genres per year
plt.figure(figsize=(12, 6))
sns.countplot(data=filtered, x='top year', hue='top genre')
plt.title("Top Genres per Year (Count)")
plt.xlabel("Year")
plt.ylabel("Number of Songs")
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
#Extract the feature which effect the general impression of the song
features = ['top year', 'bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop']
corr_matrix = df_full[features].corr()
#Create a mask to only show the lower triangle to avoid repetition (and also looks cuter)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

#Set up the matplotlib figure
plt.figure(figsize=(10, 8))

#Draw the heatmap with the mask and correct aspect ratio
#Shout out to Tanish Maheshwari, Tarpara Nisarg Bhaveshbhai, and Mitali Halder! They inspired me for the first step to deal with the dataset
sns.heatmap(
    corr_matrix,
    mask=mask,                   #Hide upper triangle
    cmap='magma',                #Color theme: dark purple to light orange
    annot=True,                  #Show the correlation values
    fmt=".2f",                   #Keep only 2 decimals
    linewidths=.5,               #Lines between cells
    cbar_kws={"shrink": .5}      #Shrink color bar size
)

plt.title("Heatmap Showing Correlation Between Audio Features", fontsize=14)
plt.tight_layout()
plt.show()

#Box plot of genre-audio feature relationship
#See which genres have significant differences in these characteristics
#Such as which genre has a higher BPM value, is more danceable, has a more positive mood, etc
top_5_genres = df_full['top genre'].value_counts().head(5).index
genre_features_data = df_full[df_full['top genre'].isin(top_5_genres)]

#Select the core dimensions that will make the style difference, rather than all the features, to make the table more concise
features_to_plot = ['bpm', 'nrgy', 'dnce', 'val','pop','acous', 'spch'] 
num_features = len(features_to_plot) #Count how many features we want to plot
cols = 3
rows = math.ceil(num_features / cols) #Figure out how many rows

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    sns.boxplot(data=genre_features_data, x='top genre', y=feature, ax=axes[i])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    axes[i].set_title(f'{feature.upper()} by Genre')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j]) #Remove empty subplots

plt.tight_layout()
plt.show()

num_features = len(features)
cols = 3
rows = math.ceil(num_features / cols)

#The distribution of feature
plt.figure(figsize=(20, 15))

features.remove ('top year') #Remove top year subplots
for i, feature in enumerate(features, 1):
    plt.subplot(rows, cols, i)
    df_full[feature].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'{feature.upper()} Distribution')
    plt.xlabel(feature.upper())
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
#%%
#Text analysis
#Import pacakages
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

#Download stopwords from nltk
nltk.download('stopwords')

#Drop missing lyrics
all_lyrics = df_full["lyrics"].dropna().str.cat(sep=" ").lower()

#Tokenize using regex: keep only words with 2+ characters, extracting words using regular expressions
tokens = re.findall(r'\b[a-z]{2,}\b', all_lyrics)

#Default and custom stopwords
stop_words = set(stopwords.words("english"))
custom_stopwords = {
    "oh", "yeah", "uh", "na", "la", "woah", "hey", "ha", "oooh", "mmm",
    "yeah", "yeahyeah", "ooh", "uhh", "aah", "yea", "yeaah", 
    "come", "on", "gotta", "gonna", "wanna", "ya", "uhuh", "whoa", 
    "ah", "ay", "coz", "cuz", "uhhuh", "get", "go", "make", "got", "let",
    "woo","cause","one","bout","eh","em","done","do","ayy","remix","album",
    "verse", "chorus", "outro", "intro", "hook","repeat", "bridge", "instrumental",
    "lyrics","huh","track","sound track","lo","de","yo","lil","eu","te","doh","da","mm",
    "ba","ich","que","nah","could","like"
}

#Filter stopwords
filtered_words = [
    word for word in tokens 
    if word not in stop_words and word not in custom_stopwords
]

#Count word frequency
word_counts = Counter(filtered_words)
top_words = word_counts.most_common(30)

#Bar chart of top words
df_counts = pd.DataFrame(top_words, columns=["word", "count"])
plt.figure(figsize=(12, 6))
plt.bar(df_counts["word"], df_counts["count"], color="orchid")
plt.xticks(rotation=45)
plt.title("Top 30 Most Frequent Words in Lyrics")
plt.tight_layout()
plt.show()

#Word cloud visualization
wordcloud = WordCloud(
    width=1000, height=500,
    background_color='white',
    colormap='viridis'
).generate_from_frequencies(word_counts)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Lyrics Word Cloud", fontsize=20)
plt.show()
#According to the result, we can the most common and meaningful words in the entire dataset
#%% 
#TF-IDF analysis, still text analysis but in a different way
#Learned the code from in textbook and the Melanie Walsh(2021), I cited in my PDF
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_lyrics(text):
    """Preprocess lyrics to make it suitable for analysis."""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to lower case and clean up
    text = str(text).lower()
    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

#Clean the lyrics
df_full['processed_lyrics'] = df_full['lyrics'].apply(preprocess_lyrics)

#Filter non-empty lyrics
non_empty_lyrics = df_full[df_full['processed_lyrics'].str.len() > 0]['processed_lyrics']

#Customize stopwords to filter meaningless single words
extended_stopwords = list(stopwords.words('english')) + [
    'oh', 'yeah', 'uh', 'na', 'la', 'woah', 'hey', 'ha', 'oooh', 'mmm',
    'yeah', 'yeahyeah', 'ooh', 'uhh', 'aah', 'yea', 'yeaah', 
    'come', 'on', 'gotta', 'gonna', 'wanna', 'ya', 'uhuh', 'whoa', 
    'ah', 'ay', 'coz', 'cuz', 'uhhuh', 'get', 'go', 'make', 'got', 'let',
    'woo', 'cause', 'one', 'bout', 'eh', 'em', 'done', 'do', 'ayy', 'remix', 'album',
    'verse', 'chorus', 'outro', 'intro', 'hook', 'repeat', 'bridge', 'instrumental',
    'lyrics', 'huh', 'track', 'sound track', 'lo', 'de', 'yo', 'lil', 'eu', 'te', 'doh', 'da', 'mm',
    'ba', 'ich', 'que', 'nah', 'could', 'like'
]

#TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(
    stop_words=extended_stopwords,
    max_features=500,
    min_df=2,  #A word must appear in at least 2 documents
    max_df=0.8,  #Ignore words that appear in more than 80% of documents
    token_pattern=r'\b[a-zA-Z]{3,}\b',  #At least 3 letters
    ngram_range=(1, 2)  #Extract single words and adjacent phrases
)

#Find the most representative and discriminative word by averaging the TF-IDF score of a word in all lyrics
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_lyrics)
    
    #Get feature names and mean scores
    feature_names = tfidf_vectorizer.get_feature_names_out()
    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    #Create TF-IDF data frame
    tfidf_scores = pd.DataFrame({
        'term': feature_names,
        'tfidf_score': mean_scores
    }).sort_values('tfidf_score', ascending=False)
    
    print("Top 20 TF-IDF Terms:")
    print(tfidf_scores.head(20))
    
    #Visualize top TF-IDF terms
    top_tfidf = tfidf_scores.head(25)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(top_tfidf)), top_tfidf['tfidf_score'], 
                    color='darkseagreen', alpha=0.8)
    plt.yticks(range(len(top_tfidf)), top_tfidf['term'])
    plt.xlabel('Average TF-IDF Score')
    plt.title('Top 25 Terms by TF-IDF Score', fontsize=16)
    plt.gca().invert_yaxis()
    
    #Add value labels for better readability
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"TF-IDF error: {e}") #Prevent errors from interrupting the entire analysis process


def analyze_yearly_tfidf_comparison():
    """Analyze TF-IDF comparison across different years."""
    df_lyrics_tfidf = df_full.dropna(subset=['processed_lyrics'])
    
    #Calculate overall TF-IDF (all songs as corpus)
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=extended_stopwords,
        max_features=500,
        min_df=3,  # Must appear in at least 3 songs
        max_df=0.7,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        ngram_range=(1, 2)
    )
    
    #Calculate TF-IDF for all lyrics
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_lyrics_tfidf['processed_lyrics'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    #Create TF-IDF DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=feature_names,
        index=df_lyrics_tfidf.index
    )
    
    #Add year information
    tfidf_df['year'] = df_lyrics_tfidf['top year']
    
    #Calculate mean TF-IDF scores for each year
    yearly_tfidf_means = tfidf_df.groupby('year').mean()
    
    return yearly_tfidf_means, feature_names


#Analyze yearly TF-IDF comparison
yearly_means, features = analyze_yearly_tfidf_comparison()

#Find most characteristic words for each year
for year in sorted(yearly_means.index):
    top_terms = yearly_means.loc[year].nlargest(10)
    print(f"\nMost characteristic words for {year}:")
    for term, score in top_terms.items():
        if score > 0:  # Only show meaningful scores
            print(f"  {term}: {score:.4f}")

top_terms = yearly_means.mean().nlargest(20).index
heatmap_data = yearly_means[top_terms]

#Visualize the popularity changes of the top 20 keywords in the past 10 years to observe the popular trends more intuitively
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", linewidths=0.5)
plt.title('Top 20 Keywords TF-IDF by Year')
plt.xlabel('Terms')
plt.ylabel('Year')
plt.tight_layout()
plt.show()
#TF-IDF was used to extract the most recognizable keywords in the lyrics of the whole and each year, reflecting the evergreens in the music content and the different trends over time. 
#For example, the frequent appearance of "love" and "tonight" in the lyrics of 2011 showed a typical romantic tendency
#while more vulgar language such as "bitch" and "fuck" emerged in 2017, indicating that underground and more aggressive lyrics dominated the mainstream that year.
#%% 
#PCA on Audio Features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# List of audio feature columns
features = ['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch']

#Scale the features
df_features = df_full[features]
df_features_scaled = StandardScaler().fit_transform(df_features) #Standardlize

#PCA to 2 dimensions
#I reduced the original 9-dimensional feature space to 2 dimensions using PCA 
#Although more principal components could retain additional variance, the first two components were sufficient to capture the dominant variance patterns and enabled intuitive visualization of clustering structures
#This dimensionality was chosen to balance interpretability and analytical utility.
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_features_scaled)

#Store PCA result in original dataframe
df_full['PC1'] = pca_result[:, 0]
df_full['PC2'] = pca_result[:, 1]

#Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_full, x='PC1', y='PC2', hue='top year', palette='tab10')
plt.title("PCA of Song Features Colored by Year")
plt.tight_layout()
plt.show()

#Print loading of features
df_pca = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
print(df_pca.T.sort_values("PC1", ascending=False))
#%%
#Cluster analysis, use the result of PCA to process K mean cluster
#Import needed packages
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#Define optimal number of clusters by silhouette score
silhouette_scores = []
K_range = range(2, 11)

#Under different cluster numbers (k), the dimensionality reduced data is clustered and its silhouette score is calculated
#Finally the k with the best performance is selected
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_result)
    silhouette_avg = silhouette_score(pca_result, cluster_labels)
    silhouette_scores.append(silhouette_avg)

#Draw the silhouette coefficient plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#According to the figure, the silhouette score is between 0.33 and 0.38, which means it is suitable for cluster analysis
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

#According to the optimal clustering
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

df_full['cluster'] = kmeans_optimal.fit_predict(pca_result)

#Visualiza clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_full['PC1'], df_full['PC2'], 
                     c=df_full['cluster'], cmap='tab10', 
                     alpha=0.7, s=50)

#Centerlize clusters
centers = kmeans_optimal.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], 
           c='blue', marker='o', s=300, linewidths=3, label='Centroids')

#Clustering feature analysis
print("Clustering feature analysis:")
cluster_summary = df_full.groupby('cluster')[features].mean()
print(cluster_summary.round(2))
#Cluster 0 is the more “fast-paced, active, loud, emotional, electronic” songs
#Cluster 1 is the more “slow-paced, quiet, emotional, acoustic, and possibly more lyrical” songs.
#%%
#Brieflly look at the year distribution in each cluster

cluster_year_counts = pd.crosstab(df_full['cluster'], df_full['top year'])
print(cluster_year_counts)


cluster_year_counts.T.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Distribution of Clusters Across Years')
plt.xlabel('Top Year')
plt.ylabel('Number of Songs')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
#%%
#In my previous statistics course, we mentioned the chi test and ANOVA test
#So I googled the code to do chi and ANOVA by SciPy, also cited in my PDF
#To see if there are significant differences between the two clusters in different dimensions

from scipy.stats import chi2_contingency

#Statistical test: see if the cluster is associated with the top year
chi2, p, dof, expected = chi2_contingency(cluster_year_counts)

print(f"Chi-square Statistic = {chi2:.2f}")
print(f"p-value (real) = {p:.10f}")

if p < 0.05:
    print("Significant association between cluster and top year.")
else:
    print("No significant association between cluster and top year.")
#Songs from different years have significant differences in audio feature clustering
#The p-value proves that the correlation is very strong and it is almost impossible to be generated randomly
#Cluster analysis successfully captures the time trend


from scipy.stats import f_oneway

anova_results = []

for feature in features:
    group0 = df_full[df_full['cluster'] == 0][feature]
    group1 = df_full[df_full['cluster'] == 1][feature]
    stat, p = f_oneway(group0, group1)
    anova_results.append({'Feature': feature, 'F-statistic': stat, 'p-value': p})

df_anova = pd.DataFrame(anova_results).sort_values(by='p-value')
print(df_anova)
#All features showed statistically significant differences (p < 0.05)
#Energy (F = 1313.03, p < 0.001), loudness (F = 704.92, p < 0.001), and acousticness (F = 310.96, p < 0.001) exhibiting the most substantial differences
#This suggests that the clustering captures meaningful stylistic divisions, with one group characterized by higher energy and loudness, and the other by more acoustic, lower-energy compositions