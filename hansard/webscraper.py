from bs4 import BeautifulSoup
import requests
import json

def fetch_hansard_report(sitting_date):
    url = 'https://sprs.parl.gov.sg/search/getHansardReport/'
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Origin': 'https://sprs.parl.gov.sg',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    
    params = {'sittingDate': sitting_date}
    
    data = {
        "headers": {
            "normalizedNames": {},
            "lazyUpdate": None
        }
    }
    
    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=200)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request for {sitting_date}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response for {sitting_date}: {e}")
        return None

def check_schema(data):
    if not data:
        return False
    
    required_fields = ['metadata', 'attendanceList', 'takesSectionVOList', 'ptbaList']
    
    for field in required_fields:
        if field not in data:
            return False
    
    metadata = data.get('metadata', {})
    metadata_fields = ['parlimentNO', 'sessionNO', 'volumeNO', 'sittingNO', 'sittingDate', 'dateToDisplay']
    
    for field in metadata_fields:
        if field not in metadata:
            return False
    
    attendance_list = data.get('attendanceList', [])
    if attendance_list:
        for mp in attendance_list:
            if 'mpName' not in mp or 'attendance' not in mp:
                return False
    
    takes_sections = data.get('takesSectionVOList', [])
    if takes_sections:
        for section in takes_sections:
            if 'title' not in section or 'content' not in section:
                return False
    
    return True

def main():
    from datetime import datetime, timedelta
    
    date_ranges = [
        ("2015-01-01", "2019-12-31", "2015-2019"),
        ("2018-05-06", "2019-12-31", "2015-2019"),
        ("2025-01-01", "2025-06-09", "2025")
    ]
    
    def date_range_generator(start_date, end_date):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)
    
    def format_date_for_api(date_obj):
        return f"{date_obj.day}-{date_obj.month}-{date_obj.year}"
    
    for start_date, end_date, folder_name in date_ranges:
        print(f"Testing date range: {start_date} to {end_date}")
        
        for date_obj in date_range_generator(start_date, end_date):
            api_date = format_date_for_api(date_obj)
            
            data = fetch_hansard_report(api_date)
            if data and check_schema(data):
                print(f"SUCCESS: {api_date}")
                # print(json.dumps(data, indent=2))

                contentList = data["takesSectionVOList"]
                for section in contentList:
                    content = section["content"]
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    section["content"] = text
                
                with open("./hansard/hansard_sections/" + api_date + ".json", "w") as f:
                    json.dump(contentList, f, ensure_ascii=False, indent=2)
                continue
            else:
                continue

main()