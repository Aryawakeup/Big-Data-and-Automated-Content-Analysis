# Spotify Top 100 Songs Analysis (2010-2019)

A comprehensive data analysis project exploring musical trends, audio features, and lyrical patterns in Spotify's top 100 songs from 2010-2019.

## Project Overview

This project combines audio feature analysis with natural language processing to uncover trends in popular music over a 5-year period. The analysis includes genre distribution, audio feature correlations, lyrical content analysis, and clustering techniques to identify distinct musical styles.

## Key Features

- **Audio Feature Analysis**: Correlation analysis of BPM, energy, danceability, valence, and other Spotify audio features
- **Genre Trend Analysis**: Evolution of music genres over time
- **Lyrics Analysis**: 
  - Word frequency analysis and word clouds
  - TF-IDF analysis to identify characteristic terms by year
  - Text preprocessing and sentiment indicators
- **Clustering Analysis**: K-means clustering with PCA dimensionality reduction
- **Statistical Testing**: ANOVA and Chi-square tests for cluster validation

## Dataset

The project uses the [Spotify Top 100 Songs 2010-2019] dataset from Kaggle, enhanced with lyrics scraped from Genius.com.

### Data Sources:
- **Primary**: Kaggle dataset with audio features and metadata
- **Secondary**: Lyrics scraped using the Genius API (lyricsgenius library)

## Technologies & Libraries

### Core Analysis
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning and clustering

### Visualization
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualizations

### Text Analysis
- `nltk` - Natural language processing
- `wordcloud` - Word cloud generation
- `sklearn.feature_extraction.text` - TF-IDF vectorization

### Data Collection
- `kagglehub` - Kaggle dataset download
- `lyricsgenius` - Genius API for lyrics scraping

### Statistical Analysis
- `scipy.stats` - Statistical testing (ANOVA, Chi-square)

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd spotify-analysis
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud kagglehub lyricsgenius scipy
   ```

3. **Download NLTK stopwords**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Set up Genius API**
   - Get API token from [Genius API](https://genius.com/api-clients)
   - Replace the token in the code: `genius = lyricsgenius.Genius("YOUR_TOKEN_HERE")`

## Analysis Pipeline

### 1. Data Collection & Cleaning
- Download Spotify dataset from Kaggle
- Scrape lyrics using Genius API
- Clean song titles and handle duplicates
- Merge datasets and handle missing values

### 2. Exploratory Data Analysis
- Genre diversity analysis by year
- Audio feature distribution analysis
- Correlation heatmap of audio features
- Box plots comparing genres

### 3. Text Analysis
- Lyrics preprocessing and tokenization
- Word frequency analysis and visualization
- TF-IDF analysis for characteristic terms
- Yearly trend analysis of lyrical content

### 4. Dimensionality Reduction & Clustering
- PCA to reduce 9 audio features to 2 dimensions
- K-means clustering with optimal k selection
- Silhouette analysis for cluster validation
- Statistical testing (ANOVA, Chi-square)

## Key Findings

### Audio Features
- **Strong correlations** between energy and loudness (r=0.72)
- **Genre differences**: Electronic music shows higher energy and danceability
- **Temporal trends**: Certain audio characteristics evolved over the 5-year period

### Lyrical Analysis
- **Common themes**: Love, relationships, and nightlife dominate across all years
- **Yearly variations**: 2011 showed romantic tendencies, while 2017 featured more aggressive language
- **TF-IDF insights**: Identified characteristic terms for each year

### Clustering Results
- **Two distinct clusters** identified:
  - **Cluster 0**: High-energy, loud, electronic songs
  - **Cluster 1**: Acoustic, lower-energy, more lyrical compositions
- **Statistical significance**: ANOVA confirms meaningful differences between clusters (p < 0.001)

## File Structure

```
├── final project5.py                   # Main analysis script
├── lyrics_result.csv            # Scraped lyrics data
├── cleaned_result.csv           # Final cleaned dataset
├── README.md                    # This file
└── visualizations/              # Generated plots and charts
```

## Important Notes

### API Limitations
- **Lyrics scraping takes ~2.5 hours** for the full dataset
- **Rate limiting**: Genius API has request limits
- **Matching challenges**: Some songs may not be found due to title variations

### Data Quality
- **99%+ matching rate** achieved for lyrics
- **Special handling** for problematic titles (encoding issues, special characters)
- **Duplicate handling** through index-based merging

## Reproducibility

1. **First run**: Execute all cells to download data and scrape lyrics
2. **Subsequent runs**: Uncomment lines to load pre-saved CSV files:
   ```python
   df_full = pd.read_csv("cleaned_result.csv")
   df_lyrics = pd.read_csv("lyrics_result.csv")
   ```

## Usage Examples

### Basic Analysis
```python
# Load cleaned data
df_full = pd.read_csv("cleaned_result.csv")

# Quick genre analysis
genre_counts = df_full['top genre'].value_counts()
print(genre_counts.head())

# Audio feature correlation
features = ['bpm', 'nrgy', 'dnce', 'val']
correlation = df_full[features].corr()
```

### Clustering Analysis
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare features
features = ['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch']
X_scaled = StandardScaler().fit_transform(df_full[features])

# Apply clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional analysis techniques
- Visualization improvements
- Code optimization
- Documentation enhancements

## Contact

For questions or collaboration opportunities, please reach out through GitHub issues.

## References

- [Genius API Documentation](https://docs.genius.com/)
- [Original Kaggle Dataset](https://www.kaggle.com/datasets/muhmores/spotify-top-100-songs-of-20152019)

---
