import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity


def merge_dataframes(filename_movies, filename_movies_title):
    column_names = ["User_ID", "User_Names", "Movie_ID", "Rating", "Timestamp"]
    movies_df = pd.read_csv(filename_movies, sep=",", names=column_names)

    # Load the move information in a DataFrame:
    movies_title_df = pd.read_csv(filename_movies_title)
    movies_title_df.rename(
        columns={"item_id": "Movie_ID", "title": "Movie_Title"}, inplace=True
    )

    # Merge the DataFrames:
    movies_df = pd.merge(movies_df, movies_title_df, on="Movie_ID")
    return movies_df, movies_title_df


def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    # Get the location of the actual user in the User-Items matrix
    # Use it to index the User similarity matrix
    user_similarities = X_user[user]
    # obtain the indices of the top k most similar users
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    # Obtain the mean ratings of those users for all movies
    rec_movies = (
        user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    )
    # Discard already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    # return recommendations - top similar users rated movies
    rec_movies_a = rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: "Movie_ID"}, inplace=True)
    return rec_movies_a


def movie_recommender_run(
    movies_df, ratings_df, rating_cosine_similarity, user_Name, movies_title_df
):
    # Get ID from Name
    user_ID = movies_df.loc[movies_df["User_Names"] == user_Name].User_ID.values[0]
    # Call the function
    temp = movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec = temp.merge(movies_title_df, how="inner")
    return top_k_rec


def main():
    filename_movies = "Movie_data.csv"
    filename_movie_titles = "Movie_Id_Titles.csv"

    movies_df, movies_title_df = merge_dataframes(
        filename_movies, filename_movie_titles
    )

    # View the DataFrame:
    print(f"\n Size of the movie_df dataset is {movies_df.shape}")

    movies_df.groupby("User_ID")["Rating"].count().sort_values(ascending=True).head()

    n_users = movies_df.User_ID.unique().shape[0]
    n_movies = movies_df.Movie_ID.unique().shape[0]

    ratings = np.zeros((n_users, n_movies))
    for row in movies_df.itertuples():
        ratings[row[1], row[3] - 1] = row[4]

    # View the matrix
    print(ratings)
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= ratings.shape[0] * ratings.shape[1]
    sparsity *= 100
    print(sparsity)
    rating_cosine_similarity = cosine_similarity(ratings)

    # Converting the 2D array into a DataFrame as expected by the movie_recommender function
    ratings_df = pd.DataFrame(ratings)
    user_ID = 12
    print(movie_recommender(ratings_df, rating_cosine_similarity, user_ID))

    # Set page configuration
    st.set_page_config(
        layout="wide", page_title="Movie Recommendation App", page_icon=":Cinema:"
    )

    # Write code to call movie_recommender_run and display recommendations
    # Read the dataset to find unique users
    column_names = ["User_ID", "User_Names", "Movie_ID", "Rating", "Timestamp"]
    movies_df = pd.read_csv("Movie_data.csv", sep=",", names=column_names)
    n_users = movies_df.User_Names.unique()

    # Create application's header
    st.header("Collaborative Filtering Recommendation System")

    # Create a dropdown of UserIDs
    User_Name = st.selectbox("Select a user name:", (n_users))

    st.write("This user might be interested in the following movies:")
    # Find and display recommendations for selected users
    result = movie_recommender_run(
        movies_df=movies_df,
        ratings_df=ratings_df,
        rating_cosine_similarity=rating_cosine_similarity,
        user_Name=User_Name,
        movies_title_df=movies_title_df,
    )
    st.table(result.Movie_Title)

    # Display movie rating charts here
    # Display details of provided recommendations
    ids = result.Movie_ID
    Names = result.Movie_Title
    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=(Names),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
    )

    # x_row and y_col will determine the location of a plot in the plot-grid
    x_row = 1
    y_col = 1
    for i in range(len(result)):
        temp = (
            (movies_df.loc[movies_df["Movie_ID"] == ids[i]])
            .groupby("Rating")
            .User_ID.count()
            .reset_index()
        )

        Rating = temp.Rating.to_numpy()
        User_ID = temp.User_ID.to_numpy()

        x_row = int(i / 2 + 1)
        y_col = i % 2 + 1

        fig.add_trace(go.Bar(x=[1, 2, 3, 4, 5], y=User_ID), row=x_row, col=y_col)

    fig.update_layout(
        height=900, width=800, showlegend=False, title="Ratings of Suggested Movies"
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
