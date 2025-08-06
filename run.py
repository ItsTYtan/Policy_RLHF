import subprocess
import sys
from hansard.webscraper import scrapeNewHansard

# scrape new hansard
newHansard = scrapeNewHansard()
print("new hansard dates: \n" + newHansard)

subprocess.run([sys.executable, "process_hansard.py"])

