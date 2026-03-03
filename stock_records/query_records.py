import sqlite3
import argparse
import os
import sys

# 用来给股票做笔记的功能
# Get DB path relative to script location
DB_PATH = os.path.join(os.path.dirname(__file__), 'analysis_records.db')

def query_records(stock_code=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        if stock_code:
            print(f"--- Analysis Records for Stock: {stock_code} ---")
            cursor.execute('''
                SELECT stock_code, record_date_display, author, content, record_date_sort
                FROM analysis_records
                WHERE stock_code = ?
                ORDER BY record_date_sort DESC, id DESC
            ''', (stock_code,))
        else:
            print("--- Latest 5 Analysis Records ---")
            cursor.execute('''
                SELECT stock_code, record_date_display, author, content, record_date_sort
                FROM analysis_records
                ORDER BY id DESC
                LIMIT 5
            ''')
            
        rows = cursor.fetchall()
        
        if not rows:
            print("No records found.")
            return

        for row in rows:
            # row: 0=code, 1=date_display, 2=author, 3=content, 4=date_sort
            print("-" * 40)
            print(f"Stock: {row[0]}")
            print(f"Date:  {row[1]}")
            print(f"Author: {row[2]}")
            print(f"Content: {row[3]}")
            
        print("-" * 40)
            
    except Exception as e:
        print(f"Error querying records: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query stock analysis records.")
    parser.add_argument("code", nargs='?', help="Optional: Stock code (6 digits) to filter by. If omitted, shows latest 5 records.")
    
    args = parser.parse_args()
    
    # If user provides a positional argument, use it as code
    # But wait, user requirement says: "Direct execution can display latest 5 records, input stock 6 digit code, can print all records..."
    # This might mean: run script -> display 5 -> ask for input -> display specific stock.
    # Let's support CLI arg first. If no CLI arg, show 5, THEN ask for input.
    
    if args.code:
        query_records(args.code)
    else:
        query_records() # Show latest 5
        
        # Interactive part
        while True:
            print("\nEnter stock code (6 digits) to query specific stock, or 'q' to quit:")
            choice = input("> ").strip()
            
            if choice.lower() == 'q':
                break
                
            if len(choice) == 6 and choice.isdigit():
                query_records(choice)
            elif choice:
                print("Invalid code. Must be 6 digits.")
