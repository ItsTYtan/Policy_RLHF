import json
import os

from bs4 import BeautifulSoup

dirPath = "/home/tytan216/volume/tzeyoung/Policy_RLHF/hansard/hansard_raw/"
files = os.listdir(dirPath)
mps = set()

for file in files:
    with open(dirPath + file, "r") as f:
        data = json.load(f)
        contentList = data["takesSectionVOList"]
        mpList = data["ptbaList"]
        for section in contentList:
            content = section["content"]
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            section["content"] = text
        for mp in mpList:
            if mp["mpName"]:
                mps.add(mp["mpName"])

        with open("./hansard/hansard_sections/" + file, "w") as f:
            json.dump(contentList, f, ensure_ascii=False, indent=2)

with open("./hansard/mps.json", "w") as f:
    json.dump(sorted(mps), f, ensure_ascii=False, indent=2)