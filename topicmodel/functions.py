import sqlite3
from bertopic import BERTopic

def list_to_tuple_str(lst):
    tup = tuple(lst)
    s = str(tup)
    # drop the comma on single-item tuples
    if len(lst) == 1:
        s = s.replace(",)", ")")
    return s

def init_topic_model(savePath):
    conn = sqlite3.connect("db/axiom.db")
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


def update_topic_model(newHansardDates, savePath):
    datesTuple = list_to_tuple_str(newHansardDates)
    conn = sqlite3.connect("db/axiom.db")
    cursor = conn.cursor()

    cursor.execute(f'''
        SELECT speech, summary
        FROM speeches s
        WHERE date IN {datesTuple}
        ORDER BY date
    ''')

    data = cursor.fetchall()
    conn.close()

    speeches = []
    summary = []
    for entry in data:
        speeches.append(entry[0])
        summary.append(entry[1])
    
    newModel = BERTopic()
    newModel.fit_transform(summary)

    base = BERTopic.load(savePath)
    merged_model = BERTopic.merge_models([base, newModel])

    merged_model.save(savePath, serialization="safetensors", save_ctfidf=True)

    baseTopics = base.get_topic_info().itertuples(index=True)
    mergedTopics = merged_model.get_topic_info().itertuples(index=True)

    assert len(baseTopics) <= len(mergedTopics)

    changedTopics = []
    newTopics = []
    for i in range(len(mergedTopics)):
        if baseTopics[i].name == mergedTopics[i].name and baseTopics[i].count != mergedTopics[i].count:
            changedTopics.append(baseTopics[i].name)
        if i > len(baseTopics):
            newTopics.append(mergedTopics[i].name)
    
    return changedTopics, newTopics
         

def get_topic_model(savePath):
    return BERTopic.load(savePath)

init_topic_model("topicmodel/model")