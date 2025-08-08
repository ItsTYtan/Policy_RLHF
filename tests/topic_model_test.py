from pathlib import Path
import sys
import pytest

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
from bertopic import BERTopic
from datasets import load_dataset
from topicmodel.functions import init_topic_model, update_topic_model, get_topic_model


def create_new_mockup():
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

    # Extract abstracts to train on and corresponding titles
    abstracts = dataset["abstract"]
    new = BERTopic()
    new.fit_transform(abstracts)

    return new


def test_update_new_model(tmp_path):
    init_topic_model(tmp_path)
    base = get_topic_model(tmp_path)
    new = create_new_mockup()

    init_topic_model(tmp_path)
    update_topic_model(new, tmp_path)
    merged = get_topic_model(tmp_path)

    baseTopics = base.get_topic_info()
    mergedTopics = merged.get_topic_info()

    assert len(baseTopics) < len(mergedTopics)

if __name__ == "__main__":
    pytest.main()