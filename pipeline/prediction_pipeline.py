from config.paths_config import *
from utils.helpers import *
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def hybrid_recommendation(user_id, user_weight=0.5, content_weight=0.5):
    """
    Generates a hybrid anime recommendation list based on user similarity and content similarity.

    Args:
        user_id (int): ID of the user for whom recommendations are to be generated.
        user_weight (float, optional): Weight for user-based recommendations. Defaults to 0.5.
        content_weight (float, optional): Weight for content-based recommendations. Defaults to 0.5.

    Returns:
        list: Top 10 recommended anime names sorted by combined hybrid scores.
    """
    try:
        ## User Recommndation

        similar_users =find_similar_users(user_id,USER_WEIGHTS_PATH,ENCODED_USER,DECODED_USER)
        user_pref = get_user_preferences(user_id,RATING_DF, ANIME_DF)
        user_recommended_animes =get_user_recommendations(similar_users,user_pref,ANIME_DF, SYNOPSIS_DF,RATING_DF)
        

        user_recommended_anime_list = user_recommended_animes["anime_name"].tolist()

        #### Content recommendation
        content_recommended_animes = []

        for anime in user_recommended_anime_list:
            similar_animes = find_similar_animes(anime, ANIME_WEIGHTS_PATH, ENCODED_ANIME, DECODED_ANIME, ANIME_DF, ANIMESYNOPSIS_CSV)

            if similar_animes is not None and not similar_animes.empty:
                content_recommended_animes.extend(similar_animes["name"].tolist())
            else:
                print(f"No similar anime found {anime}")
        
        combined_scores = {}

        for anime in user_recommended_anime_list:
            combined_scores[anime] = combined_scores.get(anime,0) + user_weight

        for anime in content_recommended_animes:
            combined_scores[anime] = combined_scores.get(anime,0) + content_weight  

        sorted_animes = sorted(combined_scores.items() , key=lambda x:x[1] , reverse=True)

        return [anime for anime , _ in sorted_animes[:10]] 
    
    except Exception as e:
        logger.error(f"Error occurred during making predictions: {e}")
        raise CustomException("Error occurred during making predictions", sys) 

