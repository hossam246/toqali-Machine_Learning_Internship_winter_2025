from datetime import datetime
formats = ["%m/%d/%Y", "%Y/%m/%d","%m-%d-%Y", "%Y-%m-%d"]
def extract_month(date_str):
    for fmt in formats:
        try:
            date_object = datetime.strptime(date_str, fmt)
            return date_object.month
        except ValueError:
            pass
    return None  

def extract_year(date_str):
    for fmt in formats:
        try:
            date_object = datetime.strptime(date_str, fmt)
            return date_object.year
        except ValueError:
            pass
    return None  