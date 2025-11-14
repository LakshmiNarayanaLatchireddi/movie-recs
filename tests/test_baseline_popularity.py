import pandas as pd
from recommender.baseline import PopularityModel

def test_popularity_fit_and_score():
    df = pd.DataFrame({
        "userId": [1, 1, 2, 2, 2],
        "movieId": [10, 11, 10, 12, 12],
        "rating": [4, 5, 3, 5, 4],
    })
    model = PopularityModel().fit(df, "userId", "movieId", "rating")
    scores = model.score_items(1, [10, 11, 12, 13])
    assert len(scores) == 4
    assert scores[0] >= scores[-1]