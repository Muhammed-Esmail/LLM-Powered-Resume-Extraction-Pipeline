#!/usr/bin/env python
# coding: utf-8

# # LLM-Powered Resume Extraction Pipeline

# ## Import Libraries

# In[336]:


# !sudo apt install tesseract-ocr
import pytesseract
from PIL import Image
import io
import pymupdf
import json
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import re
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import List

load_dotenv()

print("Imported All Libraries.")


# ## Configuration

# In[337]:


MODEL_NAME = 'llama-3.1-8b-instant'
OCR_METHOD = 'tesseract'


# ## OCR Logic

# In[338]:


def ocr_image_tesseract(image):
    text = pytesseract.image_to_string(image,lang='eng')
    return text


# In[339]:


OCR_METHOD_MAP = {
    'tesseract': ocr_image_tesseract
}


# In[340]:


def ocr_image(image):
    method = OCR_METHOD_MAP[OCR_METHOD]
    return method(image)


# ## PDF Text Extraction

# In[341]:


def get_pdf_text(pdf_file_path):
    pages = []

    try:
        doc = pymupdf.open(pdf_file_path)
    except Exception as e:
        print(f"[PDF Text Extractor] Error with provided pdf file path: {e}")
        return

    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < 50:
            print(f"[PDF Text Extractor] Page #{i+1} likely an image. Running OCR...")

            pix = page.get_pixmap(dpi=300)

            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_image(img)
            print(f"[PDF Text Extractor] Page #{i+1} OCR Done!")
        else:
            print(f"[PDF Text Extractor] Page #{i+1} is text based.")

        pages.append(text)

    return pages


# ## LLM Management

# In[342]:


def prompt_builder(resume_text):
    return f'''
    You are a professional resume reviewer and analyzer. You are reviewing a resume text and are required to extract select information
    from the given text into a structured JSON format.

    RESUME TEXT: [{resume_text}]
    RESUME TEXT ENDED.

    You are required to extract these pieces of information from the resume:
        - Resume Holder's Name
        - Resume Holder's Email
        - Resume Holder's Phone number
        - Resume Holder's Skills
        - Resume Holder's Work Experiences
        - Resume Holder's Educations


    Caution:
        - Be aware of the input text. Text possibly contains typos, missing letters or missing words. Do your best to write the output free of typos modifying text only if strictly necessary.
        This issue might be apparent in emails, skills, work experience sections and/or education. Do not alter the name/email. Only alter name/email if it is a very obvious mistake in the text.

    Output Format:
        - JSON output format:
            {{
                "Name": NAME,
                "Email": EMAIL,
                "Phone": 123-456-7890,
                "Skills": [
                    "SKILL_1",
                    "SKILL_2",
                    ...
                ],
                "Work Experience": [
                    "EXPERIENCE_1",
                    "EXPERIENCE_2",
                    ...
                ],
                "Education": [
                    "EDUCATION_1",
                    "EDUCATION_2",
                ]
            }}
        - Replace NAME, EMAIl, SKILL, EXPERIENCE and EDUCATION with the values you extract from the given resume text.
        - All property names must be wrapped in double quotes NOT single quotes.
        - Name and Email fields are strings, Phone is a string phone number formatted as xxx-xxx-xxxx, Skills, Work Experience and Education are lists of strings.
        - Always Output the Name field as Pascal Case (capitalize only first letters of each name and separate by spaces) regardless of the original case in the text.
        - It is acceptable to output empty lists for any of (Skills, Work Experience, Education) if you are not confident enough in your output.
        - Do not hallucinate. Only output text that is actually present in the provided text and only from the text. Do not make up experiences/education/skills. 
        - For each skill in the skills section, write as much of the original description as possible. Avoid single-word or short-worded skills as possible. Only add short worded skills if there were no more description for them in the provided text. Do not make up skills or descriptions not originally there.
        - If a field is not clear and/or you are not confident enough in your answer leave the field's value as null.
        - Only output a valid structured JSON file ready to be parsed. Do not output any text/opinions along with your JSON output. Do not say 'Here is ...' or any accompanying text. Only output the JSON structure: {{...}}
        - For any date you write in your output write it as: DD/MM/YYYY. Write the months explicitly as (Jan/Feb/Mar/..) etc. not as numbers. Write days and years as numbers.
    '''


# In[343]:


def ask_model_groq(prompt):
    LLM_KEY = os.getenv('GROQ_API_KEY')

    URL = 'https://api.groq.com/openai/v1/responses'
    payload = {
        'input': prompt,
        'model': MODEL_NAME
    }
    headers = {
        'Authorization': f'Bearer {LLM_KEY}',
        'Content-Type': "application/json"
    }

    response = requests.post(
        url=URL,
        json=payload,
        headers=headers
    )

    result = response.text
    result = json.loads(result)

    if 'output' not in result:

        pattern = r'Please try again in (\d+(?:\.\d+)?(ms|s))'

        if 'error' in result:

            message = result.get('error').get('message')

            print("Error: ", message)

            match = re.search(pattern, message)

            wait_time = match.group(1)

            if wait_time.endswith('s'):
                wait_time = float(wait_time[:-1])
            else:
                wait_time = float(wait_time[:-2]) *1e-3


            if wait_time:
                wait_time = float(wait_time) + 0.2
                print(f"[GROQ API] Rate Limited. Waiting {wait_time}s.")

            time.sleep(wait_time)

            raise Exception(f"Rate Limited. Ready to request again.")


        raise ValueError(f"Unexpected API response: {result}")

    return result['output'][1]['content'][0]['text']


# ## Helper Methods

# In[344]:


class ResumeSchema(BaseModel):
    Name: str
    Email: EmailStr
    Phone: str
    Skills: List[str]
    WorkExperience: List[str] = Field(alias="Work Experience")
    Education: List[str]


# In[ ]:


def extract_and_validate_json(response: str):
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        raise Exception(f"[JSON Validation] No JSON found in response.")

    try:
        raw_json = json.loads(match.group())
        return ResumeSchema(**raw_json).model_dump(by_alias=True)
    except Exception as e:
        raise Exception(f"[JSON Validation] Encountered Error: {e}")


# In[346]:


def init_df():
    df = pd.DataFrame({
        'File Name': [],
        'File Path': [],
        'Name': [],
        'Email': [],
        'Phone': [],
        'Skills': [],
        'Work Experience': [],
        'Education': []
    })
    return df


# ## Main Program

# In[347]:


def pipeline(pdfs, model_output, json_list_output = None, verbose=False):
    if not isinstance(pdfs, list):
        pdfs = [pdfs]

    df = init_df()

    for i, pdf in enumerate(tqdm(pdfs)):
        texts = get_pdf_text(pdf)

        resume_text = '\n\n'.join(texts)

        prompt = prompt_builder(resume_text)

        retry = 5
        retry_cnt = retry
        terminate = False

        while retry_cnt >= 0:

            try:
                response = ask_model_groq(prompt)

                output = extract_and_validate_json(response)

                if verbose:
                    print(f"[Pipeline] Received: {response}")
                else:
                    print(f"[Pipeline] Processing pdf #{i+1} Complete.")

                break
            except Exception as e:
                print(f"[Pipeline Attempt #{retry+1 - retry_cnt} Ran into an issue: {e}. Retrying...]")

                retry_cnt = retry_cnt - 1
                if retry_cnt < 0:
                    terminate = True
                continue

        if terminate:
            print(f"[Pipeline] Processing pdf #{i+1} Failed. Continuing...")
            continue

        new_row = pd.DataFrame([output])

        if json_list_output is not None:
            json_list_output.append(output)


        file_path = pdf

        file_name = str(pdf).split('/')[-1]

        new_row['File Path'] = str(file_path)
        new_row['File Name'] = str(file_name)

        df = pd.concat([df, new_row], ignore_index=True)


    model_output = pd.concat([model_output, df], ignore_index=True)

    return model_output


# ## Output Generation

# In[348]:


data_dir_path = os.getcwd() / Path('data/')


# In[349]:


model_output = init_df()


# In[350]:


pdfs_to_work = [data_dir_path / x for x in os.listdir(data_dir_path) if x.endswith('.pdf')]


# In[351]:


json_output_list = []


# In[352]:


model_output = pipeline(pdfs_to_work, model_output, json_list_output=json_output_list)


# In[353]:


model_output.head(10)


# In[354]:


# print(json_output_list[0])


# In[ ]:


obj = model_output.to_dict()


# In[356]:


filename = f'data/output {datetime.now()}.json'
filename


# In[ ]:


with open(filename, 'w') as file:
    json.dump(obj, file, ensure_ascii=False, indent=4)

