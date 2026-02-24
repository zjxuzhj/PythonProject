import sqlite3
import argparse
import datetime
import os
import sys

# Get DB path relative to script location
DB_PATH = os.path.join(os.path.dirname(__file__), 'analysis_records.db')

def parse_date(date_str):
    """
    Parses 'MM.DD' or 'MM-DD' or 'YYYY.MM.DD' string into sortable YYYY-MM-DD.
    If year is missing, infers it based on current date to handle near-future/past.
    However, given stock analysis context, usually it's current or recent past.
    
    Logic:
    If MM.DD is provided:
      - If month is > current month + 2, assume it's last year (e.g. input Dec in Feb).
      - Else assume current year.
    This is a heuristic.
    """
    today = datetime.date.today()
    current_year = today.year
    
    # Normalize separators
    date_str = date_str.replace('-', '.')
    
    parts = date_str.split('.')
    
    if len(parts) == 3:
        # YYYY.MM.DD
        return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    elif len(parts) == 2:
        # MM.DD
        month = int(parts[0])
        day = int(parts[1])
        
        # Simple heuristic:
        # If input month is 12 and current month is 1 or 2, likely last year.
        # If input month is 1 or 2 and current month is 12, likely next year (rare for analysis).
        
        year = current_year
        if month > today.month + 6: # e.g. input 12 in Feb -> 12 > 8, so last year
             year = current_year - 1
        elif month < today.month - 6: # e.g. input 1 in Aug -> 1 < 2, so next year? No, usually past.
             # If I'm in Dec and input is Jan, it's Jan of this year (past).
             # If I'm in Jan and input is Dec, it's Dec of last year.
             pass
             
        # More robust logic: compare distance to today
        try:
            # Candidate 1: This year
            d1 = datetime.date(current_year, month, day)
            
            # If d1 is in future by more than 1 month, assume last year
            if d1 > today + datetime.timedelta(days=30):
                year = current_year - 1
            else:
                year = current_year
        except ValueError:
            # Invalid date (e.g. Feb 30), default to current year
            year = current_year
            
        return f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    else:
        return date_str # Fallback

def add_record(stock_code, date_str, author, content):
    if len(stock_code) != 6 or not stock_code.isdigit():
        print("Error: Stock code must be 6 digits.")
        return

    sort_date = parse_date(date_str)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO analysis_records (stock_code, record_date_display, record_date_sort, author, content)
            VALUES (?, ?, ?, ?, ?)
        ''', (stock_code, date_str, sort_date, author, content))
        conn.commit()
        print("Record added successfully.")
    except Exception as e:
        print(f"Error adding record: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a stock analysis record.")
    parser.add_argument("-c", "--code", help="Stock code (6 digits)")
    parser.add_argument("-d", "--date", help="Date (MM.DD)")
    parser.add_argument("-a", "--author", default="自己", help="Author (default: 自己)")
    parser.add_argument("-m", "--message", help="Content of the record")
    
    args = parser.parse_args()

    # If any required arg is missing, enter interactive mode
    if not (args.code and args.date and args.message):
        print("Entering interactive mode...")
        
        # Ask for code if missing
        code = args.code
        while not code or len(code) != 6 or not code.isdigit():
             code = input("Stock Code (6 digits): ").strip()
             
        # Ask for date if missing
        date_in = args.date
        while not date_in:
             date_in = input("Date (MM.DD): ").strip()
             
        # Ask for author if missing (use default if enter)
        author = args.author
        if author == "自己": # Check if user wants to change default
            temp_author = input(f"Author (default: {author}): ").strip()
            if temp_author:
                author = temp_author
        
        # Ask for content if missing
        content = args.message
        while not content:
             content = input("Content: ").strip()
        
        add_record(code, date_in, author, content)
    else:
        add_record(args.code, args.date, args.author, args.message)
