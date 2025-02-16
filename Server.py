from tokenize import Token
from fastapi import FastAPI,Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os ,json
import openai
import re,requests
import uvicorn
from fastapi import FastAPI, Request, Query, Form, Body
from pydantic import BaseModel
from typing import Optional
import aiofiles
from PIL import Image
import pytesseract,cv2
import asyncio


from Functions import *

app = FastAPI()

# Allow all CORS origins (modify as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def map_task_with_openai(task_description: str):
    # Define a prompt for the OpenAI API
    prompt = f"""
    You are a task detection system. Your job is to analyze the following task description and map it to one of the following task IDs:
    -"A1: Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with $'user.email' as the only argument.'\n"
    -"A2: 'Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place'\n"
    -"A3: 'The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays and write just the number to /data/dates-wednesdays.txt'\n"
    -"A4: 'Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json'\n"
    -"A5: 'Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first'\n"
    -"A6: 'Find all Markdown (.md) files in /data/docs/, extract the first occurrence of each H1, and create an index file /data/docs/index.json mapping filenames to titles'\n"
    -"A7: '/data/email.txt contains an email message. Extract the sender’s email address using an LLM and write it to /data/email-sender.txt'\n"
    -"A8: '/data/credit-card.png contains a credit card number. Use an LLM to extract the card number and write it without spaces to /data/credit-card.txt'\n"
    -"A9: '/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line'\n"
    -"A10: 'The SQLite database file /data/ticket-sales.db has a table tickets with columns type, units, and price. Calculate the total sales for the \"Gold\" ticket type and write the number to /data/ticket-sales-gold.txt'\n\n"
    -"B7": Compress or resize an image
    -"B9": Convert Markdown to HTML

    Task Description: {task_description}

    Return only the task ID (e.g., A1, A2, etc.) that best matches the task description.
    """
    #********************************************* IMPORTANT *******************************************************************************************************#
    token = os.getenv("AIPROXY_TOKEN")
    if not token :
        raise ValueError("API_KEY IS NOT SET IN ENVIRONMENT")
    client = openai.Client(
        base_url="https://aiproxy.sanand.workers.dev/openai/v1",  # Correct way to set base URL
        api_key = token
    )
    # Call the OpenAI API
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an AI that converts natural language tasks into executable commands."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    timeout=10,  # Set timeout for API call
    max_tokens=50
    )
    print(response)
    task_id = response.choices[0].message.content.strip()
    return task_id
    #****************************************************** IMPORTANT **********************************************************************************************#
    # payload = {
    #     "model": "mistral",  # Use the model name you have installed
    #     "prompt": prompt,
    #     "temperature": 0.0,
    #     "max_tokens": 50  # Adjust as needed
    # }

    # try:
    #     response = requests.post(
    #         f"http://host.docker.internal:11434/v1/completions",  # Correct API endpoint
    #         json=payload,
    #         timeout=60
    #     )

    #     if response.status_code == 200:
    #         data = response.json()
    #         generated_text = data['choices'][0]['text'].strip()
    #         return generated_text
    #     else:
    #         raise Exception(f"Error calling Ollama: {response.status_code} - {response.text}")
    # except Exception as e:
    #     raise Exception(f"An error occurred while communicating with the Ollama API: {e}")

    # Extract the task ID from the response
  



#****************************************************** IMPORTANT **********************************************************************************************#

def call_openai(prompt):
    """Calls OpenAI API to interpret the task."""
    print('API call initiated')
    token = os.getenv("AIPROXY_TOKEN")
    if not token :
        raise ValueError("API_KEY IS NOT SET IN ENVIRONMENT")
    client = openai.Client(
        base_url="https://aiproxy.sanand.workers.dev/openai/v1",  # Correct way to set base URL
        api_key=token
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that converts natural language tasks into executable commands."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        timeout=10 , # Set timeout for API call
        max_tokens=250
    )
    print('API call completed')
    return response.choices[0].message.content.strip()



#****************************************************** IMPORTANT **********************************************************************************************#

# def call_openai(prompt):
#     """
#     Calls the local Ollama API with the given prompt and returns the generated response.
#     """
#     payload = {
#         "model": "mistral",  # Use the model name you have installed
#         "prompt": prompt,
#         "temperature": 0.0,
#         "max_tokens": 250  # Adjust as needed
#     }
#     try:
#         response = requests.post(
#             f"http://host.docker.internal:11434/v1/completions",  # Correct API endpoint
#             json=payload,
#             timeout=200
#         )
#         print(1111111111)
#         if response.status_code == 200:
#             data = response.json()
#             generated_text = data['choices'][0]['text'].strip()
#             return generated_text
#         else:
#             raise Exception(f"Error calling Ollama: {response.status_code} - {response.text}")
#     except Exception as e:
#         raise Exception(f"An error occurred while communicating with the Ollama API: {e}")
##################################################################################################################################################################
def is_path_allowed(path):
    """
    Ensure the path is within the /data directory.
    """
    # Resolve the absolute path
    absolute_path = Path(path).resolve()
    # Define the allowed base directory
    allowed_base = Path("/data").resolve()
    # Check if the path is within the allowed base directory
    return allowed_base in absolute_path.parents or absolute_path == allowed_base



def passes_luhn(card_number):
    """ Luhn algorithm to validate credit card numbers. """
    digits = [int(d) for d in card_number]
    checksum = 0
    double = False

    for i in range(len(digits) - 1, -1, -1):
        d = digits[i]
        if double:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
        double = not double

    return checksum % 10 == 0

def preprocess_image(input_file):
    """ Preprocess the image to improve OCR accuracy. """
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize for clarity
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply thresholding
    return img

def extract_credit_card_number(input_file):
    try:
        # 1. Preprocess the image
        img = preprocess_image(input_file)

        # 2. Convert processed image to PIL format for Tesseract
        img = Image.fromarray(img)

        # 3. Configure Tesseract for digits only
        custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
        extracted_text = pytesseract.image_to_string(img, config=custom_config)

        # 4. Extract lines, look for a line with exactly 16 digits
        lines = extracted_text.splitlines()
        recognized_16 = None
        for line in lines:
            digits = re.sub(r"\D", "", line)  # Keep only digits
            if len(digits) == 16:
                recognized_16 = digits
                break

        if not recognized_16:
            return {
                "error": "No line with exactly 16 digits found.",
                "output": extracted_text
            }

        # 5. Check Luhn
        if passes_luhn(recognized_16):
            return {"output": recognized_16}
        else:
            # If first digit is '9', try flipping it to '3'
            if recognized_16[0] == '9':
                possible_fix = '3' + recognized_16[1:]
                if passes_luhn(possible_fix):
                    return {"output": possible_fix}
                else:
                    return {
                        "error": "Luhn check failed, flipping '9'->'3' also failed.",
                        "output": recognized_16
                    }
            else:
                return {
                    "error": "Luhn check failed and no known fix.",
                    "output": recognized_16
                }

    except Exception as e:
        return {"error": str(e)}



@app.get('/')
def home():
    return 'Application Running Successfully !! '





class TaskRequest(BaseModel):
    task: str

@app.post("/run")
async def run_task(
    request: Request,
    task: Optional[str] = Query(None),  # Accept task as query parameter
    form_task: Optional[str] = Form(None),  # Accept task from form data
    body_task: Optional[TaskRequest] = Body(None),  # Accept task from JSON body
):
    print("Entered function")

    # Determine the task from available sources
    task_text = (
        (body_task.task if body_task else None) or
        (form_task if form_task else None) or
        (task if task else None)
    )

    if not task_text:
        return {"status": "error", "message": "Task is missing"}

    decoded_task = task_text.strip().replace("\n", " ").replace("`", "")
    task_id = map_task_with_openai(decoded_task)
    print('*'*50)
    print(task_id)
    if (task_id== "A1"):

        # url_pattern = r"(https?://[^\s]+)"
        # email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        # url_match = re.search(url_pattern, decoded_task)
        # email_match = re.search(email_pattern, decoded_task)
        # script_url = url_match.group(0) if url_match else None
        # email = email_match.group(0) if email_match else None

        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:

            - **Package Name**: The package required for the task.
            - **Script URL**: The URL of the script that needs to be downloaded and executed.
            - **Email Id Argument**: The Email-Id that have to be passed as an argument.

            Task Description: {decoded_task}

            Return the details **strictly** as a JSON object with the following keys:
            {{
            "package_name": "Name of the package to install",
            "script_url": "URL of the script to execute",
            "email_id": "Email-Id that have to be passed to the script"
            }}

            Do not include any explanations, just return the JSON object.
            """
        ai_response = json.loads(call_openai(prompt))
        package,script_url,email_id = str(ai_response.get("package_name")),str(ai_response.get("script_url")),str(ai_response.get("email_id"))
        print(script_url,email_id)
        return task_a1(user_url=script_url,user_email=email_id)


    elif (task_id== "A2"):
        prompt = f"""
            You are a task automation assistant. Your job is to extract the following details from the task description:
            - **Input File**: The file to read from.

            Task Description:
            {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.

            {{
                "input_file": "Input file name or path",
            }}

            Do not include any explanations, just return the JSON object.
            """
        ai_response=json.loads(call_openai(prompt))
        input_file=ai_response.get("input_file")
        if not is_path_allowed(input_file) :
            raise ValueError("Access to paths outside /data is not allowed.")
        return task_a2(input_file)


    elif (task_id== "A3"):
        prompt = f"""
            You are a task automation assistant. Your job is to extract the following details from the task description:
            - Input File: The file to read from.
            - Output File: The file to write to.
            Task Description: {decoded_task}

                Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
                1. The output is **not wrapped in quotes, triple quotes, or backticks**.
                2. The output is **pure JSON**, without any additional text.

                {{
                "input_file": "Input file name or path",
                "output_file": "Output file name or path"
                }}

                Do not include any explanations, just return the JSON object"""
        ai_response = json.loads(call_openai(prompt))
        input_file , output_file = ai_response.get("input_file"),ai_response.get("output_file")
        string = decoded_task.lower()
        day_tuple = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday")
        for _ in day_tuple:
            if _ in string :
                day = _
        print(day,input_file,output_file)
        if not is_path_allowed(input_file) or not is_path_allowed(output_file):
            raise ValueError("Access to paths outside /data is not allowed.")
        return task_a3(day,input_path=input_file,output_path=output_file)


    elif (task_id== "A4"):
        prompt = prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:
            - Input File: The file to read from.
            - Output File: The file to write to.
            - Sorting Criteria: The attributes used for sorting and their order.

            Task Description: {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.

            {{
            "input_file": "Input file name or path",
            "output_file": "Output file name or path",
            "sorting_criteria": ["Primary sorting attribute", "Secondary sorting attribute",]
            }}

            Ensure that the sorting criteria are listed in order of priority. Do not include any explanations, just return the JSON object.
            """
        ai_response = json.loads(call_openai(prompt))
        input_file , output_file ,sorting_style = ai_response.get("input_file"),ai_response.get("output_file"),ai_response.get("sorting_criteria")
        if not is_path_allowed(input_file) or not is_path_allowed(output_file):
            raise ValueError("Access to paths outside /data is not allowed.")
        return task_a4(input_file,output_file,sorting_style)


    elif (task_id== "A5"):
        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:
            - Input Path: The directory or file(s) to read from.
            - Output Path: The file to write to.
            - Number of Files: The number of files to process.
            - Extension Type: The file extension to filter by.
            - Extraction Rule: The specific condition used to extract data.

            Task Description: {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.

            {{
            "input_file": "Directory or file path for reading",
            "output_file": "File path for writing",
            "number_of_files": "Number of files to process",
            "extension_type": "File extension to filter (e.g., .log, .txt)",
            }}

            Do not include any explanations, just return the JSON object.
            """
        ai_response = json.loads(call_openai(prompt))
        print(ai_response)
        
        input_file , output_file ,extension_type , numberoffiles = ai_response.get("input_file"),ai_response.get("output_file"),str(ai_response.get("extension_type")),int(ai_response.get("number_of_files"))
        if not is_path_allowed(input_file) or not is_path_allowed(output_file):
            raise ValueError("Access to paths outside /data is not allowed.")

        return task_a5(input_file,output_file,extension_type,numberoffiles)

        
    elif (task_id== "A6"):
        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:
            - Input Path: The directory or file(s) to read from.
            - Output Path: The file to write to.
            - Extension Type: The file extension to filter by.

            Task Description: {decoded_task}

           Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.

            {{
            "input_file": "Input file name or path",
            "output_file": "Output file name or path",
            "extension_type: "File extension to filter (e.g., .md, .log, .txt)
            }}
            Do not include any explanations, just return the JSON object.
            """
        ai_response = json.loads(call_openai(prompt))
        print("IN Task A6 ")
        print(ai_response)
        input_file , output_file ,extension_type = ai_response.get("input_file"),ai_response.get("output_file"),str(ai_response.get("extension_type"))
        print(input_file,output_file,extension_type)
        if not is_path_allowed(input_file) or not is_path_allowed(output_file):
            raise ValueError("Access to paths outside /data is not allowed.")
        return task_a6(input_file,output_file,extension_type)
    

    
    elif (task_id== "A7"):
        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:

            - Input Path : Path of the input file containing the email message.
            - Output Path: Path where the extracted email address should be written.

            Task Description: {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.

            {{
                "input_file": "Input file name or path",
                "output_file": "Output file name or path",
            }}
            Do not include any explanations, just return the JSON object.
            
            """
        ai_response=json.loads(call_openai(prompt))
        print(ai_response)
        input_file , output_file =ai_response.get("input_file"),ai_response.get("output_file")
        print(input_file , output_file)
        if not is_path_allowed(input_file) or not is_path_allowed(output_file):
            raise ValueError("Access to paths outside /data is not allowed.")
        with open(input_file, "r") as file:
             email_content = file.read()
        print(email_content)
        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:
                - "email": The extracted sender's email address.
            
            Task Description: {email_content}


            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.
            {{
                "email": "extracted_email_here"
            }} 
            Do not include any explanations, just return the JSON object.
          
        """
        ai_response=json.loads(call_openai(prompt))
        sender_email=ai_response.get("email")
        print('#####################################')
        print(sender_email)
        return task_a7(sender_email,output_file)


    elif (task_id== "A8"):
        prompt = f"""
            You are an advanced task automation assistant. Extract the following details from the task description:
            - Input Image: The image file containing the credit card number.
            - Output Path: The file to write the extracted result to.

            Task Description: {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.
            {{
            "input_image": "Path to the image file containing the credit card number",
            "output_path": "File path to write the result to"
            }}

            Do not include any explanations, just return the JSON object.
            """
        ai_response = json.loads(call_openai(prompt))
        print(ai_response)    
        input_path,output_path = ai_response.get("input_image"),ai_response.get("output_path")
        cardnum = (extract_credit_card_number(input_path))
        print(cardnum)
        credit_cardnumber = re.search(r"^\d+",cardnum["output"]).group()

        return task_a8(credit_cardnumber,output_path)



    elif (task_id=="A9"):
        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:

            - Input Path : Path of the input file .
            - Output Path: Path to which the output will be written.

            Task Description: {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.

            {{
                "input_file": "Input file name or path",
                "output_file": "Output file name or path",
            }}
            Do not include any explanations, just return the JSON object.
            
            """
        ai_response = json.loads(call_openai(prompt))
        print(ai_response)    
        input_path,output_path = ai_response.get("input_file"),ai_response.get("output_file")
        print(input_path,output_path)
        return await task_a9(input_path,output_path)




    elif (task_id== "A10"):
        prompt = f"""
            You are a task automation assistant. Extract the following details from the task description:
            - Database File: The SQLite database file to read from.
            - SQL Query: The correct SQL query to perform the calculation.
            - Output Path: The file to write the result to.

            Task Description: {decoded_task}

            Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
            1. The output is **not wrapped in quotes, triple quotes, or backticks**.
            2. The output is **pure JSON**, without any additional text.
            {{
            "database_file": "Path to the SQLite database file",
            "sql_query": "The exact SQL query to perform the required calculation",
            "output_path": "File path to write the result to"
            }}

            Do not include any explanations, just return the JSON object.
            """
        ai_response = json.loads(call_openai(prompt))
        input_path,output_path,calculation = ai_response.get("database_file"),ai_response.get("output_path"),str(ai_response.get("sql_query"))
        if not is_path_allowed(input_path) or not is_path_allowed(output_path):
            raise ValueError("Access to paths outside /data is not allowed.")
        print(calculation)

        return task_a10(input_path,output_path,calculation)


    # elif (task_id == "B3") or ("B3" in task_id) :
    #     prompt = f"""
    #     You are a task automation assistant. Your job is to extract the following details from the task description:

    #         - API URL: The API endpoint to fetch data from.
    #         - Output File: The file path where the fetched data should be saved.

    #         Task Description: {decoded_task}

    #         Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
    #         1. The output is **not wrapped in quotes, triple quotes, or backticks**.
    #         2. The output is **pure JSON**, without any additional text.

    #         Example Output:
    #         {{
    #         "api_url": "API endpoint URL",
    #         "output_file": "Output file name or path"
    #         }}

    #         Do not include any explanations, just return the JSON object."""
    #     ai_response = json.loads(call_openai(prompt))
    #     url,output_path = ai_response.get("api_url"),ai_response.get("output_file")
    #     if not is_path_allowed(output_path):
    #         raise ValueError("Access to paths outside /data is not allowed.")
    #     return task_b3(url,output_path)




    # elif (task_id == "B5") or ("B5" in task_id):
    #     prompt = f"""
    #     You are a task automation assistant. Your job is to extract the following details from the task description:

    #         - API URL: The API endpoint to fetch data from.
    #         - Output File: The file path where the fetched data should be saved.

    #         Task Description: {decoded_task}

    #         Return the details **strictly as a valid JSON object**, without any extra formatting, explanations, or code blocks. Ensure that:
    #         1. The output is **not wrapped in quotes, triple quotes, or backticks**.
    #         2. The output is **pure JSON**, without any additional text.

    #         Example Output:
    #         {{
    #         "db_path": "Path to the Database File",
    #         "sql_query": "The exact SQL query to perform the required calculation",
    #         "output_file": "Output file name or path"
    #         }}

    #         Do not include any explanations, just return the JSON object."""
    #     ai_response = json.loads(call_openai(prompt))
    #     input_path,output_path,calculation = ai_response.get("database_file"),ai_response.get("output_path"),str(ai_response.get("sql_query"))
    #     if not is_path_allowed(input_path) or not is_path_allowed(output_path):
    #         raise ValueError("Access to paths outside /data is not allowed.")
    #     print(calculation)
    #         return task_a10(input_path,output_path,calculation)
    #     else:
    #         raise Exception("Method Not Allowed") 





    elif (task_id == "B7") or ("B7" in task_id):
        prompt = f"""
        You are a task automation assistant. Your job is to extract the following details from the task description:

            Input File: The file path of the image to be processed.
            Output File: The file path where the compressed or resized image should be saved.
            Target Size (Optional): If specified, the target width and height for resizing.
            Compression Quality (Optional): If specified, the quality level for compression (0-100).
            Task Description: {decoded_task}

            Return the details strictly as a valid JSON object, without any extra formatting, explanations, or code blocks. Ensure that:

            1.The output is not wrapped in quotes, triple quotes, or backticks.
            2.The output is pure JSON, without any additional text.

            Example Output:
            {{
            "input_file": " Path to Input image",
            "output_file": "Path to Output Image",
            "target_size": [width, height],
            "compression_quality": quality_value
            }}

            Do not include any explanations, just return the JSON object."""


        ai_response = json.loads(call_openai(prompt))
        input_path,output_path,target_size,quality=ai_response.get("input_file"),ai_response.get("output_file"),ai_response.get("target_size"),ai_response.get("compression_quality")
        if not is_path_allowed(input_path) or not is_path_allowed(output_path):
            raise ValueError("Access to paths outside /data is not allowed.")
        return task_b7(input_path,output_path,target_size,quality)



    elif (task_id == "B9") or ("B9" in task_id):
        prompt = f"""
        "You are a task automation assistant. Your job is to extract the following details from the task description:

            -Input File: The file path of the Markdown file to be converted.
            -Output File: The file path where the converted HTML file should be saved.
            Task Description: {decoded_task}

            Return the details strictly as a valid JSON object, without any extra formatting, explanations, or code blocks. Ensure that:

            The output is not wrapped in quotes, triple quotes, or backticks.
            The output is pure JSON, without any additional text.
            Example Output:
            {
            "input_file": "path/to/input.md",
            "output_file": "path/to/output.html"
            }

            Do not include any explanations, just return the JSON object."
        """
        ai_response = json.loads(call_openai(prompt))
        input_path,output_path=ai_response.get("input_file"),ai_response.get("output_file")
        if not is_path_allowed(input_path) or not is_path_allowed(output_path):
            raise ValueError("Access to paths outside /data is not allowed.")
        return task_b9(input_path,output_path)



class PathRequest(BaseModel):
    path: str

@app.get("/read")
async def read_file(
    request: Request,
    path: Optional[str] = Query(None),  # Accept path as query parameter
    form_path: Optional[str] = Form(None),  # Accept path from form data
    body_path: Optional[PathRequest] = Body(None),  # Accept path from JSON body
):
    # Determine the path from available sources
    file_path = (
        (body_path.path if body_path else None) or
        (form_path if form_path else None) or
        (path if path else None)
    )

    if not file_path or not os.path.exists(file_path):
        return {"status": "error", "message": "File not found"}
    # Read file asynchronously
    try:
        async with aiofiles.open(file_path, 'rb') as file:
            content = await file.read()
        text_content=content.decode("utf-8",errors="ignore")
        # Return as plain text, preserving formatting
        return Response(text_content, media_type="text/plain")  
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
