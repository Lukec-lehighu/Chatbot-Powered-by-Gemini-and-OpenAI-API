import os
import google.generativeai as gen_ai
from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm # for progress reporting

#load .env file
load_dotenv()

#define constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATASET_PATH = "ACM HEALTH Datasets_V1 - CBT.csv"
COLUMN_READ_COUNT = 20

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config = generation_config
)

#start chat session
chat_session = model.start_chat(history=[])

#open csv file
df = pd.read_csv(DATASET_PATH)
responses = []
for index, row in tqdm(df.iterrows()):
    if index >= COLUMN_READ_COUNT:
        break

    prompt = row.iloc[2]

    resp_row = {}
    try:
        resp_row['prompt'] = prompt
        resp_row['normal'] = model.generate_content(contents=prompt).text
        resp_row['hispanic-male-sensitivity'] = model.generate_content(contents=prompt + ". Respond with sensitivity to Hispanic culture. The speaker is male").text
        resp_row['hispanic-female-sensitivity'] = model.generate_content(contents=prompt + ". Respond with sensitivity to Hispanic culture. The speaker is female").text
        resp_row['asian-male-sensitivity'] = model.generate_content(contents=prompt + ". Respond with sensitivity to Asian culture. The speaker is male").text
        resp_row['asian-female-sensitivity'] = model.generate_content(contents=prompt + ". Respond with sensitivity to Asian culture. The speaker is female").text
    except Exception as e:
        print(e)
        break

    responses.append(resp_row)

pd.DataFrame(responses).to_csv('gemini_output.csv')
