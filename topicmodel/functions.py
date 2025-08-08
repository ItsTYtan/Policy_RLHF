import sqlite3
from bertopic import BERTopic

def init_topic_model(savePath):
    conn = sqlite3.connect("db/axiom.db")  # Creates or opens the database file
    cursor = conn.cursor()

    cursor.execute('''
        SELECT speech, summary
        FROM speeches s
        ORDER BY date
    ''')

    data = cursor.fetchall()
    conn.close()

    speeches = []
    summary = []
    for entry in data:
        speeches.append(entry[0])
        summary.append(entry[1])

    base = BERTopic()
    base.fit_transform(summary)
    base.save(savePath, serialization="safetensors", save_ctfidf=True)


def update_topic_model(newModel, savePath):
    base = BERTopic.load(savePath)
    merged_model = BERTopic.merge_models([base, newModel])

    print("old model: " + base.get_topic_info())
    print("new model: " + merged_model.get_topic_info())

    merged_model.save(savePath, serialization="safetensors", save_ctfidf=True)

def get_topic_model(savePath):
    return BERTopic.load(savePath)