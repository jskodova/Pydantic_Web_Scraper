import datetime
import pandas as pd
from httpx import Client
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior
from load_models import OPENAI_MODEL, OLLAMA_MODEL

class Product(BaseModel):
    brand_name: str = Field(title='Brand name', description='The brand name of the product')
    product_name: str = Field(title='Product name', description='The name of the product')
    price: str | None = Field(title='Price', description='The price of the product') 
    rating_count: str | None = Field(title='Rating count', description='The rating count of the product')
    
class Results(BaseModel):
    dataset: list[Product] = Field(title='Dataset', description='The list of scraped products')

web_scraping_agent = Agent(
name='Web scraping agent',
model=OPENAI_MODEL,
system_prompt=(
    """
    Your task is to convert a data string scraped from the internet into a lost of dictionaries.

    Step 1. Fetch the HTML text from the given URL using the function fetch_html_text()
    Step 2. Take the output from text 2 and clean it up for the final output
    """
),
retries=1,
result_type=Results,
model_settings=ModelSettings(
    max_tokens=4000,
    temperature=0.1
)
)

@web_scraping_agent.tool_plain(retries=1)
def fetch_html_text(url: str) -> str:
    """
    Fetches the HTML text from the provided URL
    
    args: 
        url: str - The page's URL to fetch the HTML text from
        
    returns:
        str: The HTML text retrieved from the webpage 
    """
    print('URL Call: ', url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64), AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    with Client(headers=headers) as client:
        response = client.get(url, timeout=200)
        if response.status_code != 200:
            return f'Failed to fetch HTML text from {url}. Status code: {response.status_code}'
        soup = BeautifulSoup(response.text, 'html.parser')
        with open('soup.txt', 'w', encoding='utf-8') as f:
            f.write(soup.get_text())
        print('Soup file saved.')
        return soup.get_text().replace('\n', '').replace('\r', '')

@web_scraping_agent.result_validator
def validate_scraping_result(result: Results) -> Results:
    print('Validating...')
    if isinstance(result, Results):
        print('Validation succesful')
        return result
    print('Validation failed')
    return None

def main() -> None:
    prompt = 'https://www.ikea.com/fi/en/cat/best-sellers/'

    try:
        response = web_scraping_agent.run_sync(prompt)
        if response.data is None:
            raise UnexpectedModelBehavior("No data could be retrieved.")
            return None
        print('-' * 50)
        print('Input tokens', response.usage().request_tokens)
        print('Output tokens', response.usage().response_tokens)
        print('Total tokens', response.usage().total_tokens)

        lst = []
        for item in response.data.dataset:
            lst.append(item.model_dump())
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        df = pd.DataFrame(lst)
        df.to_csv(f'product_listings_{timestamp}.csv', index='false')
    except UnexpectedModelBehavior as e:
        print(e)

main()