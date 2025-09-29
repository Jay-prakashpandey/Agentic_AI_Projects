import os
import requests
from google.cloud import bigquery
from dotenv import load_dotenv

# Load keys
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Init BigQuery client
bq_client = bigquery.Client()

MODEL = "x-ai/grok-4-fast:free"

def query_bigquery(sql: str):
    """Run SQL on BigQuery and return result as dataframe"""
    query_job = bq_client.query(sql)
    return query_job.to_dataframe()

def call_llm(messages):
    """Call Grok model from OpenRouter"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def analyst_chat(user_question):
    """Agentic loop: Convert question → SQL → Query BigQuery → Summarize"""
    
    system = {
        "role": "system",
        "content":'''
        You are a data analyst agent. Translate user questions into BigQuery SQL for the dataset `bigquery-public-data.thelook_ecommerce`. 
        - **Return ONLY the SQL Do NOT prefix with `SQL:`, `Query:`, `sql `, or triple quotes. example: SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders`** 
        - these are list of tables distribution_centers, events, inventory_items, order_items, orders, products, thelook_ecommerce-table and users'''
    }
    user = {"role": "user", "content": user_question}

    sql = call_llm([system, user])
    
    try:
        df = query_bigquery(sql)
        summary = call_llm([
            {"role": "system", "content": "You are a data analyst. Summarize query results in plain English"},
            {"role": "user", "content": f"SQL:\n{sql}\n\nResults:\n{df.to_dict()}"}
        ])
        return {"sql": sql, "data": df.head(10), "summary": summary}
    except Exception as e:
        return {"sql": sql, "error": str(e)}
    
if __name__ == "__main__":
    sql='''SELECT table_name
FROM `bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.TABLES`
WHERE table_type = 'BASE TABLE'
ORDER BY table_name;'''
    df = query_bigquery(sql)
    print(df)