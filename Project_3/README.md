# Project 3: Movie Recommendation System (AI & Personalization)

## 📌 Project Overview
This project builds a recommendation engine similar to those used by Netflix or YouTube. It helps users discover movies by finding items that are mathematically similar to the ones they already like.

## ⚙️ Technical Approach: Content-Based Filtering
For this implementation, I focused on **Content-Based Filtering**. 
* **Logic:** Instead of looking at what other people liked, the system looks at the attributes (genres) of the movie itself.
* **Math:** I utilized **Cosine Similarity** to calculate the distance between different movie genre vectors. Movies with a smaller "angle" between their vectors are considered more similar.



## 🛠️ Tools & Libraries
* **Python 3.14**
* **Pandas:** For data manipulation and handling Tab-Separated Values (TSV).
* **Scikit-learn:** Specifically the `cosine_similarity` metric.
* **Dataset:** MovieLens 100K (Benchmark dataset in AI Research).

## 📊 Sample Output
When I tested the system with a movie like *Toy Story (1995)*, the engine successfully identified other high-similarity family and animation films:

### 📊 Project Execution & Results
Below is the screenshot of the recommendation engine in action, showing the top 5 movie suggestions for 'Toy Story' and 'Star Wars':

![Movie Recommendation Output](movie_recommendation_output.png)

## 🚀 Key Learning
Through this project, I learned how to handle non-standard data formats (pipe-separated and tab-separated) and how to transform categorical text data into a numerical matrix that an AI model can understand.
