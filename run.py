import subprocess
import sys
from hansard.webscraper import scrapeNewHansard
from db.update_db import update_db
from topicmodel.functions import update_topic_model

# scrape and process new hansard
newHansardDates = scrapeNewHansard()
print("new hansard dates: \n" + newHansardDates)
subprocess.run([sys.executable, "process_hansard.py"])

# update db with new hansard
update_db(newHansardDates, "qwen/qwen-2.5-72b-instruct")

# update topic model
changedTopics, newTopics = update_topic_model(newHansardDates, "topicmodel/model")
print("changed topics: ", changedTopics)
print("new topics: ", newTopics)

