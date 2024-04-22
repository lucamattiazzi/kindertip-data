import pandas as pd
from openai import OpenAI

URL = "http://localhost:1234/v1"
RECIPE_MODEL = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
KEY = "lm-studio"

meals = pd.read_parquet("data/meals.parquet")
client = OpenAI(base_url=URL, api_key=KEY)

def generate_recipe(row):
  completion = client.chat.completions.create(
    model=RECIPE_MODEL,
    messages=[
      {"role": "system", "content": "Sei un cuoco, partendo dal nome di un piatto risponderai con una breve descrizione del piatto in una singola frase in italiano. Nel caso si trattasse di un piatto molto semplice, come le carote, una semplice descrizione sarà sufficiente."},
      {"role": "user", "content": row["meal"]}
    ],
    temperature=0.7,
  )
  content = completion.choices[0].message.content
  first_part = content.split("\n")[0]
  return first_part

def ingredients_list(row):
  completion = client.chat.completions.create(
    model=RECIPE_MODEL,
    messages=[
      {"role": "system", "content": "Sei un cuoco, partendo dal nome di un piatto risponderai con SOLAMENTE l'elenco degli ingredienti necessari per preparare il piatto in italiano. Gli ingredienti saranno separati da virgole."},
      {"role": "user", "content": row["meal"]}
    ],
    temperature=0.7,
  )
  content = completion.choices[0].message.content
  first_part = content.split("\n")[0]
  return first_part

def get_taste(row):
  completion = client.chat.completions.create(
    model=RECIPE_MODEL,
    messages=[
      {"role": "system", "content": "Sei un cuoco, partendo dal nome di un piatto risponderai con SOLAMENTE quali dei 5 gusti principali (dolce, salato, amaro, acido, umami) sono presenti nel piatto. Se non sei sicuro, rispondi con 'non lo so'."},
      {"role": "user", "content": row["meal"]}
    ],
    temperature=0.7,
  )
  content = completion.choices[0].message.content
  first_part = content.split("\n")[0]
  return first_part

def get_embedding(row, column):
   return client.embeddings.create(input = [row[column]], model=EMBEDDING_MODEL).data[0].embedding


meals["recipe"] = meals.apply(generate_recipe, axis=1)
meals.to_parquet("data/meals_with_recipes.parquet")
meals["ingredients"] = meals.apply(ingredients_list, axis=1)
meals.to_parquet("data/meals_with_recipes.parquet")
meals["taste"] = meals.apply(get_taste, axis=1)
meals.to_parquet("data/meals_with_recipes.parquet")
meals["recipe_embeddings"] = meals.apply(lambda row: get_embedding(row, "recipe"), axis=1)
meals.to_parquet("data/meals_with_recipes.parquet")
meals["meal_embeddings"] = meals.apply(lambda row: get_embedding(row, "meal"), axis=1)
meals.to_parquet("data/meals_with_recipes.parquet")
