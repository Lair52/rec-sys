import pandas as pd
from predict_template import TopicPredictor, TopicPredictionRequest


def sucess_request():
    df_content = pd.read_csv(
        "./data/learning-equality-curriculum-recommendations/content.csv"
    )
    request = TopicPredictionRequest(
        content_title=str(df_content.iloc[1]["title"]),
        content_description=str(df_content.iloc[1]["description"]),
        content_text=str(df_content.iloc[1]["text"]),
    )
    return request


def fail_request():
    request = TopicPredictionRequest(
        content_title=None,
        content_description=None,
        content_text=None,
    )
    return request


def testing_predictor(request_ex):
    predictor = TopicPredictor()
    topic_ids = predictor.predict(request_ex)
    print("Predicted topic_ids:", topic_ids)
    return topic_ids  # <- importante para podermos fazer asserts


if __name__ == "__main__":
    print("SUCCESS REQUEST:")
    sucess_req = sucess_request()
    sucess_topics = testing_predictor(sucess_req)

    assert isinstance(sucess_topics, list), "Predict should return a list for valid input."
    assert len(sucess_topics) > 0, "Predict should return at least one topic_id for valid input."
    assert all(isinstance(t, str) for t in sucess_topics), "All topic_ids should be strings."

    print("\nFAIL REQUEST:")
    fail_req = fail_request()
    try:
        fail_topics = testing_predictor(fail_req)

        assert isinstance(fail_topics, list), "Predict should return a list even for empty input."
        print("Fail-case handled without errors.")
    except Exception as e:
        print("Fail-case raised an exception (this is acceptable if you chose to enforce non-empty input):")
        print(repr(e))
