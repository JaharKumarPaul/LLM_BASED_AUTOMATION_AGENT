import os
import subprocess,glob
import json
from datetime import datetime
from pathlib import Path
import sqlite3,requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
import markdown
import httpx
import asyncio

# Task A1: Install uv (if required) and run datagen.py with ${user.email} as the argument
def task_a1(user_url:str="https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py",user_email: str = '21f3000512@ds.study.iitm.ac.in',package_name:str="uv"):
    # Install uv if not already installed
    print('In Task A1')
    try:
        subprocess.run([package_name, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        subprocess.run(["pip", "install", package_name], check=True)

    response = requests.get(user_url)
    if response.status_code == 200:
        script_content = response.text
        
        # Save it to a file
        with open("datagen.py", "w") as file:
            file.write(script_content)
        
        # Run the script with the email argument
        subprocess.run(["python3", "datagen.py", user_email], check=True)
    else:
        print("Failed to download script.")




# Task A2: Format /data/format.md using prettier@3.4.2
def task_a2(file_path):
    with open(file_path,'r') as f :
        original = f.read()
    subprocess.run(
        ["prettier", "--write", file_path],
        input=original,
        capture_output=True,
        text=True,
        check=True,
        # Ensure npx is picked up from the PATH on Windows
        # shell=True,
    ).stdout
    print('Okaaaay')



# Task A3: Count the number of Wednesdays in /data/dates.txt
def task_a3(day:str,input_path,output_path):
    date_formats = [
        "%Y/%m/%d %H:%M:%S",  # e.g., 2008/04/22 06:26:02
        "%Y-%m-%d",           # e.g., 2006-07-21
        "%b %d, %Y",          # e.g., Sep 11, 2006
        "%d-%b-%Y",           # e.g., 28-Nov-2021
    ]

    day_count = 0

    with open(input_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parsed_date = None
            # Try each date format until one succeeds.
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(line, fmt)
                    break  # Exit loop if parsing is successful.
                except ValueError:
                    continue

            if parsed_date is None:
                # Optionally log the unparsable line.
                print(f"Warning: Could not parse date: {line}")
                continue

            # datetime.weekday() returns Monday=0, Tuesday=1, Wednesday=2, etc.
            Day_dict={"monday":0, "tuesday":1, "wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}

            if parsed_date.weekday() == Day_dict[day]:
                day_count += 1

    with open(output_path, "w") as file:
        file.write(str(day_count))





# Task A4: Sort /data/contacts.json by last_name, then first_name
def task_a4(input_path,output_path,sorting_style):
    with open(input_path, "r") as file:
        contacts = json.load(file)

    contacts_sorted = sorted(contacts, key=lambda x: (x[sorting_style[0]], x[sorting_style[1]]))

    with open(output_path, "w") as file:
        json.dump(contacts_sorted, file, indent=2)





# Task A5: Write the first line of the 10 most recent .log files in /data/logs/
def task_a5(input_path,output_path,extension_type,files_count:int):
    print("In task a5")
    print(input_path,output_path,extension_type,files_count)
    # Get all .log files in the directory
    log_files = glob.glob(os.path.join(rf"{input_path}", f'*{extension_type}'))

    # Sort files by modification time (most recent first)
    log_files.sort(key=os.path.getmtime, reverse=True)

    # Open the output file for writing
    with open(output_path, 'w') as outfile:
        # Process the 10 most recent files
        for log_file in log_files[:files_count]:
            with open(log_file, 'r') as infile:
                first_line = infile.readline().strip()  # Read the first line
                # Write the filename and first line to the output file
                outfile.write(f"{first_line}\n")



# Task A6: Create an index of Markdown files in /data/docs/
def task_a6(input_path,output_path,extension):
    print("In function A6")
    index = {}
    for md_file in Path(input_path).rglob(f"*{extension}"):
        with open(md_file, "r") as file:
            for line in file:
                if line.startswith("# "):
                    index[str(md_file.relative_to(input_path))] = line.strip("# ").strip()
                    break

    with open(output_path, "w") as file:
        json.dump(index, file, indent=2)
    print("Succesfully Finished A6")


# Task A7: Extract the sender's email address from /data/email.txt using an LLM
def task_a7(sender_email,output_path):
    print(sender_email,output_path)
    with open(output_path, "w") as file:
        file.write(sender_email)



# Task A8: Extract a credit card number from /data/credit-card.png using an LLM
def task_a8(credit_card_number,output_path):
    # Simulate LLM interaction (replace with actual LLM API call)
    with open(output_path, "w") as file:
        file.write(credit_card_number)
    print("Successfully Completed A8")




# Task A9: Find the most similar pair of comments in /data/comments.txt
import asyncio

async def task_a9(input_path, output_path):
    print("In Task A9")
    token = os.getenv("AIPROXY_TOKEN")
    if not token :
        raise ValueError("API_KEY IS NOT SET IN ENVIRONMENT")

    openai_api_base = "http://aiproxy.sanand.workers.dev/openai/v1"
    openai_api_key = token  # Replace with actual API key

    # Read comments from the input file
    with open(input_path, 'r') as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    data = comments  # Input data for API

    # Compute embeddings for all comments asynchronously
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{openai_api_base}/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": data},
        )

    response_json = response.json()
    
    embeddings = np.array([emb["embedding"] for emb in response_json["data"]])

    # Compute similarity matrix
    similarity = np.dot(embeddings, embeddings.T)

    # Ignore self-similarity
    np.fill_diagonal(similarity, -np.inf)

    # Find the most similar pair
    i, j = np.unravel_index(np.argmax(similarity), similarity.shape)

    # ✅ FIX: Use correct variable for output
    most_similar_comments = "\n".join(sorted([data[i], data[j]]))

    # Write to output file
    with open(output_path, 'w') as f:
        f.write(most_similar_comments)

    print("Successfully Completed A9")




# Task A10: Calculate total sales for "Gold" ticket types in /data/ticket-sales.db
def task_a10(input_path,output_path,calculation:str):
    conn = sqlite3.connect(input_path)
    cursor = conn.cursor()

    cursor.execute(calculation)
    total_sales = cursor.fetchone()[0]

    with open(output_path, "w") as file:
        file.write(str(total_sales))

    conn.close()

# Main function to execute all tasks


# def task_b3(url,output_path):
#     try:
#         # Fetch data from the API
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an error for bad status codes

#         # Save the data to the specified file
#         with open(output_path, "wb") as file:
#             file.write(response.content)
#         print(f"Data successfully saved to {output_path}")

#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data from the API: {e}")
#     except IOError as e:
#         print(f"Error saving data to the file: {e}")


# def task_b5(db_path, query, output_filename):
#     """
#     Run a SQL query on a SQLite or DuckDB database and save the results to a file.

#     :param db_path: Path to the database file (SQLite or DuckDB).
#     :param query: SQL query to execute.
#     :param output_filename: Output file name with the desired extension.
#     :return: Query results as a list of rows."""

#     import sqlite3, duckdb

#     # Connect to the database
#     if db_path.endswith('.db'):
#         conn = sqlite3.connect(db_path)
#     else:
#         conn = duckdb.connect(db_path)

#     cur = conn.cursor()
#     cur.execute(query)
#     result = cur.fetchall()
#     conn.close()

#     # Save the results to the output file
#     with open(output_filename, 'w') as file:
#         # Handle different file extensions
#         if output_filename.endswith('.csv'):
#             import csv
#             csv_writer = csv.writer(file)
#             csv_writer.writerows(result)
#         elif output_filename.endswith('.json'):
#             import json
#             json.dump(result, file, indent=4)
#         elif output_filename.endswith('.txt'):
#             for row in result:
#                 file.write(str(row) + '\n')
#         elif output_filename.endswith('.db'):
#             # Save to a new SQLite database
#             import sqlite3
#             new_conn = sqlite3.connect(output_filename)
#             new_cur = new_conn.cursor()
#             new_cur.execute("CREATE TABLE results AS SELECT * FROM (VALUES (?, ?))")  # Example table creation
#             new_cur.executemany("INSERT INTO results VALUES (?, ?)", result)
#             new_conn.commit()
#             new_conn.close()
#         else:
#             # Default: Save as plain text
#             file.write(str(result))

#     return result



def task_b7(input_file,output_file,target_size=None,compression_quality=50):
    try:
        # Open the image
        with Image.open(input_file) as img:
            # Resize if target size is provided
            if target_size:
                img = img.resize(target_size, Image.ANTIALIAS)

            # Save with specified compression quality
            img.save(output_file, "JPEG", quality=compression_quality)
            print(f"Image successfully processed and saved to {output_file}")

    except Exception as e:
        print(f"Error processing image: {e}")


def task_b9(input_file, output_file):
    """
    Convert a Markdown file to an HTML file.

    :param input_file: Path to the input Markdown file.
    :param output_file: Path to save the converted HTML file.
    """
    try:
        # Read Markdown content
        with open(input_file, "r", encoding="utf-8") as md_file:
            md_content = md_file.read()
        
        # Convert Markdown to HTML
        html_content = markdown.markdown(md_content)
        
        # Save the HTML output
        with open(output_file, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
        
        print(f"Markdown successfully converted to HTML and saved to {output_file}")

    except Exception as e:
        print(f"Error converting Markdown to HTML: {e}")

