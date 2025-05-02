import pandas as pd
import numpy as np
import joblib
import sys
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def getAnimeFrame(anime, path_df):
    """
    Retrieve metadata information for a specific anime from a dataset.

    Args:
        anime (int or str): The anime identifier, either as an anime ID (int) or English title (str).
        path_df (str): File path to the anime metadata CSV.

    Returns:
        pd.DataFrame or np.nan: A DataFrame row containing the anime's metadata if found, otherwise NaN.
    """

    try:
        df = pd.read_csv(path_df)
        if isinstance(anime, int):
            result = df[df.anime_id == anime]
        elif isinstance(anime, str):
            result = df[df.eng_version == anime]
        else:
            raise ValueError("Anime must be an int (anime_id) or str (eng_version)")

        if result.empty:
            return np.nan
        return result

    except Exception as e:
        logger.error("Failed in getAnimeFrame")
        raise CustomException(str(e), sys)

def getSynopsis(anime, path_synopsis_df):
    """
    Retrieve the synopsis of a given anime.

    Args:
        anime (int or str): The anime ID or English title.
        path_synopsis_df (str): File path to the synopsis dataset CSV.

    Returns:
        str or np.nan: Synopsis string if found, otherwise NaN.
    """

    try:
        synopsis_df = pd.read_csv(path_synopsis_df)
        if isinstance(anime, int):
            return synopsis_df[synopsis_df.MAL_ID == anime].sypnopsis.values[0]
        if isinstance(anime, str):
            return synopsis_df[synopsis_df.Name == anime].sypnopsis.values[0]
    except IndexError:
        return np.nan
    except Exception as e:
        logger.error("Failed in getSynopsis")
        raise CustomException(str(e), sys)

def find_similar_animes(name, path_anime_weights, path_encoded_anime, path_decoded_anime,
                        path_anime_df, path_synopsis_df, n=10, return_dist=False, neg=False):
    """
    Find anime titles similar to a given anime based on latent feature similarity.

    Args:
        name (str or int): The name or ID of the reference anime.
        path_anime_weights (str): File path to the matrix of latent anime features.
        path_encoded_anime (str): Path to dictionary mapping anime IDs to latent indices.
        path_decoded_anime (str): Path to dictionary mapping latent indices to anime IDs.
        path_anime_df (str): File path to anime metadata CSV.
        path_synopsis_df (str): File path to synopsis data CSV.
        n (int): Number of similar animes to return (default is 10).
        return_dist (bool): Whether to return similarity distances and indices (default is False).
        neg (bool): If True, retrieves least similar animes instead of most similar.

    Returns:
        pd.DataFrame or tuple: A DataFrame of similar animes, or a tuple (distances, indices) if return_dist is True.
    """

    try:
        anime_weights = joblib.load(path_anime_weights)
        encoded_anime = joblib.load(path_encoded_anime)
        decoded_anime = joblib.load(path_decoded_anime)

        anime_frame = getAnimeFrame(name, path_anime_df)
        if anime_frame is None or anime_frame.empty:
            raise ValueError(f"Anime '{name}' not found in dataset.")

        index = anime_frame.anime_id.values[0]
        encoded_index = encoded_anime.get(index)

        if encoded_index is None:
            raise ValueError(f"Encoded index not found for anime ID: {index}")

        dists = np.dot(anime_weights, anime_weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n += 1

        closest = sorted_dists[:n] if neg else sorted_dists[-n:]

        if return_dist:
            return dists, closest

        SimilarityArr = []
        for close in closest:
            decoded_id = decoded_anime.get(close)
            if decoded_id is None:
                logger.warning(f"No decoded anime found for encoded index {close}. Skipping.")
                continue

            synopsis = getSynopsis(decoded_id, path_synopsis_df)
            anime_frame = getAnimeFrame(decoded_id, path_anime_df)

            if anime_frame is None or anime_frame.empty:
                logger.warning(f"Anime ID {decoded_id} not found in metadata. Skipping.")
                continue

            anime_name = anime_frame.eng_version.values[0]
            genre = anime_frame.Genres.values[0]
            similarity = dists[close]

            SimilarityArr.append({
                "anime_id": decoded_id,
                "name": anime_name,
                "similarity": similarity,
                "genre": genre,
                "synopsis": synopsis
            })

        result_frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
        result_frame = result_frame[result_frame.anime_id != index].drop(['anime_id'], axis=1)

        return result_frame

    except Exception as e:
        logger.error("An error occurred in find_similar_animes")
        raise CustomException(str(e), sys)

def find_similar_users(user_id, path_user_weights, path_encoded_user, path_decoded_user, n=10, return_dist=False, neg=False):
    """
    Identify users similar to a given user based on collaborative filtering.

    Args:
        user_id (int): Target user ID to find similarities against.
        path_user_weights (str): File path to the matrix of latent user features.
        path_encoded_user (str): Path to dictionary mapping user IDs to latent indices.
        path_decoded_user (str): Path to dictionary mapping latent indices to user IDs.
        n (int): Number of similar users to return (default is 10).
        return_dist (bool): Whether to return similarity distances and indices.
        neg (bool): If True, returns least similar users instead of most similar.

    Returns:
        pd.DataFrame or tuple: A DataFrame of similar users with similarity scores, or tuple if return_dist is True.
    """

    try:
        user_weights = joblib.load(path_user_weights)
        encoded_user = joblib.load(path_encoded_user)
        decoded_user = joblib.load(path_decoded_user)

        encoded_index = encoded_user.get(user_id)
        if encoded_index is None:
            raise ValueError(f"User ID {user_id} not found in encoded_user mapping.")

        dists = np.dot(user_weights, user_weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n += 1

        closest = sorted_dists[:n] if neg else sorted_dists[-n:]

        if return_dist:
            return dists, closest

        similarity_arr = []
        for close in closest:
            similarity = dists[close]
            decoded_id = decoded_user.get(close)

            if decoded_id is not None and decoded_id != user_id:
                similarity_arr.append({
                    "similar_users": decoded_id,
                    "similarity": similarity
                })

        return pd.DataFrame(similarity_arr).sort_values(by="similarity", ascending=False)

    except Exception as e:
        logger.error("An error occurred in find_similar_users")
        raise CustomException(str(e), sys)

def get_user_preferences(user_id, path_rating_df, path_anime_df):
    """
    Get a user's top anime preferences based on high ratings.

    Args:
        user_id (int): The user ID whose preferences are to be retrieved.
        path_rating_df (str): File path to the user rating data.
        path_anime_df (str): File path to the anime metadata CSV.

    Returns:
        pd.DataFrame: A DataFrame containing top-rated animes and their genres.
    """

    try:
        rating_df = pd.read_csv(path_rating_df)
        df = pd.read_csv(path_anime_df)

        animes_watched = rating_df[rating_df.user_id == user_id]
        if animes_watched.empty:
            raise ValueError(f"No ratings found for user_id: {user_id}")

        rating_threshold = np.percentile(animes_watched.rating, 75)
        high_rated = animes_watched[animes_watched.rating >= rating_threshold]

        top_anime_ids = high_rated.sort_values(by="rating", ascending=False).anime_id.values
        anime_df_rows = df[df["anime_id"].isin(top_anime_ids)][["eng_version", "Genres"]]

        return anime_df_rows

    except Exception as e:
        logger.error("An error occurred in get_user_preferences")
        raise CustomException(str(e), sys)

def get_user_recommendations(similar_users, user_pref, path_anime_df, path_synopsis_df, path_rating_df, n=10):
    """
    Recommend new animes to a user based on preferences of similar users.

    Args:
        similar_users (pd.DataFrame): DataFrame containing similar user IDs.
        user_pref (pd.DataFrame): DataFrame of animes the user has already liked.
        path_anime_df (str): File path to anime metadata.
        path_synopsis_df (str): File path to synopsis data.
        path_rating_df (str): File path to user rating data.
        n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame of recommended animes with name, genre, synopsis, and user preference count.

    Raises:
        ValueError: If no similar users are provided.
        CustomException: For unexpected errors during recommendation generation.
    """

    try:
        if similar_users.empty:
            raise ValueError("No similar users found for recommendation.")

        recommended_animes = []
        anime_list = []

        for user_id in similar_users.similar_users.values:
            pref_list = get_user_preferences(int(user_id), path_rating_df, path_anime_df)
            pref_list = pref_list[~pref_list.eng_version.isin(user_pref.eng_version.values)]

            if not pref_list.empty:
                anime_list.append(pref_list.eng_version.values)

        if anime_list:
            anime_list_flat = pd.Series(np.concatenate(anime_list))
            sorted_list = anime_list_flat.value_counts().head(n)

            for anime_name, n_user_pref in sorted_list.items():
                if isinstance(anime_name, str):
                    try:
                        frame = getAnimeFrame(anime_name, path_anime_df)
                        if frame.empty:
                            continue
                        anime_id = frame.anime_id.values[0]
                        genre = frame.Genres.values[0]
                        synopsis = getSynopsis(int(anime_id), path_synopsis_df)

                        recommended_animes.append({
                            "n": n_user_pref,
                            "anime_name": anime_name,
                            "Genres": genre,
                            "Synopsis": synopsis
                        })
                    except Exception as e_inner:
                        logger.warning(f"Could not fetch details for anime '{anime_name}': {e_inner}")
                        continue

        return pd.DataFrame(recommended_animes).head(n)

    except Exception as e:
        logger.error("An error occurred in get_user_recommendations")
        raise CustomException(str(e), sys)
