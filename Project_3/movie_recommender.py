import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD DATA 
# We define the column names because the 'u.item' file doesn't have a header row
cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Loading the movie titles and their genres
movies = pd.read_csv('u.item', sep='|', names=cols, encoding='latin-1')

# 2. CREATE GENRE MATRIX
# We select only the 1s and 0s from the genre columns (columns 5 to the end)
genre_matrix = movies.iloc[:, 5:]

# 3. CALCULATE COSINE SIMILARITY
# This is the "AI" part. It measures the distance between movie genres.
cos_sim = cosine_similarity(genre_matrix, genre_matrix)

# 4. THE RECOMMENDATION FUNCTION
def recommend_movies(movie_name, top_n=5):
    try:
        # Get the index of the movie title you typed
        idx = movies[movies['title'] == movie_name].index[0]
        
        # Look up its similarity scores and sort them
        scores = list(enumerate(cos_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Take the top 5 (skipping index 0 because that's the same movie)
        top_indices = [i[0] for i in scores[1:top_n+1]]
        
        return movies['title'].iloc[top_indices]
    except IndexError:
        return "Movie not found. Make sure the name and year match exactly (e.g., 'Toy Story (1995)')"

# 5. TEST IT!
print("--- Recommendations for 'Toy Story (1995)' ---")
print(recommend_movies('Toy Story (1995)'))

print("\n--- Recommendations for 'Star Wars (1977)' ---")
print(recommend_movies('Star Wars (1977)'))
