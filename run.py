import subprocess
import sys
from hansard.webscraper import scrapeNewHansard
from db.update_db import update_db

# scrape new hansard
newHansard = scrapeNewHansard()
print("new hansard dates: \n" + newHansard)

subprocess.run([sys.executable, "process_hansard.py"])

update_db(newHansard, "qwen/qwen-2.5-72b-instruct")
