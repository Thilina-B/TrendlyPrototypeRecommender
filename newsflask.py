from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('news_list.csv')

# Replace NaN with an empty string
df['description'] = df['description'].fillna('')

# Create a TfidfVectorizer and remove stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the data to a tfidf matrix
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute the cosine similarity between each movie description
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create indices series
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(keyword, cosine_sim=cosine_sim, num_recommend=10):
    # Search for movies containing the keyword in their title or description
    movie_matches = df[df['title'].str.contains(keyword, case=False) | df['description'].str.contains(keyword, case=False)]
    
    # If there are no matches, return a message
    if movie_matches.empty:
        return "No matches found for the keyword: {}".format(keyword)

    # Get the indices of the matching movies
    indices_list = movie_matches.index.tolist()

    # Calculate the average similarity scores for the matching movies
    avg_sim_scores = cosine_sim[indices_list].mean(axis=0)

    # Sort the movies based on the average similarity scores
    movie_indices = avg_sim_scores.argsort()[::-1][:num_recommend]

    # Return the top recommended movies
    return df['title'].iloc[movie_indices].tolist()

@app.route('/recommendations', methods=['GET'])
def recommend_movies():
    keyword = request.args.get('keyword', '')
    num_recommend = int(request.args.get('num_recommend', 10))

    recommendations = get_recommendations(keyword, num_recommend=num_recommend)
    return jsonify(recommendations)

@app.route('/all_data', methods=['GET'])
def get_all_data():
    num_records = int(request.args.get('num_records', len(df)))
    all_data = df.head(num_records).to_dict(orient='records')
    return jsonify(all_data)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(
    host="0.0.0.0",
    port=5000
)
