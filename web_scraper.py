import datetime
import logging
import pandas as pd
import re
from httpx import Client, HTTPStatusError
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior
from load_models import OPENAI_MODEL, OLLAMA_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Product(BaseModel):
    brand_name: str = Field(title="Brand name", description="The brand name of the product")
    product_name: str = Field(title="Product name", description="The name of the product")
    price: str | None = Field(title="Price", description="The price of the product")
    rating_count: str | None = Field(title="Rating count", description="The rating count of the product")

class Results(BaseModel):
    dataset: list[Product] = Field(title="Dataset", description="The list of scraped products")

web_scraping_agent = Agent(
    name="Web scraping agent",
    model=OPENAI_MODEL,
    system_prompt=(
        """
        Your task is to convert a data string scraped from the internet into a list of dictionaries.

        Step 1: Fetch the HTML text from the given URL using the function fetch_html_text()
        Step 2: Process and clean the extracted text for structured output.
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
    Fetches the HTML text from the provided URL.

    Args: 
        url (str): The page's URL to fetch the HTML text from.
        
    Returns:
        str: The cleaned HTML text retrieved from the webpage.
    """
    logging.info(f"Fetching HTML from URL: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Language": "en-US,en;q=0.5"
    }
    try:
        with Client(headers=headers, timeout=20) as client:
            response = client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            clean_text = re.sub(r"[\n\r]+", " ", soup.get_text())
            return clean_text
    except HTTPStatusError as e:
        logging.error(f"Failed to fetch HTML from {url}. HTTP Error: {e}")
        return f"Error: Unable to fetch content from {url}"
    except Exception as e:
        logging.error(f"Unexpected error fetching HTML: {e}")
        return f"Error: {e}"

@web_scraping_agent.result_validator
def validate_scraping_result(result: Results) -> Results | None:
    """
    Validates the structure of the extracted data.
    
    Args:
        result (Results): The extracted product data.

    Returns:
        Results | None: The validated results or None if validation fails.
    """
    logging.info("Validating scraped data...")
    if isinstance(result, Results) and result.dataset:
        logging.info("Validation successful.")
        return result
    logging.warning("Validation failed. No valid data retrieved.")
    return None

def main() -> None:
    """
    Main function that triggers the web scraping agent, processes the data,
    and saves it to a CSV file.
    """
    prompt = "https://www.ikea.com/fi/en/cat/best-sellers/"
    
    try:
        response = web_scraping_agent.run_sync(prompt)
        
        if response.data is None or not response.data.dataset:
            raise UnexpectedModelBehavior("No valid data retrieved.")

        usage = response.usage()
        logging.info(f"Input Tokens: {usage.request_tokens}, Output Tokens: {usage.response_tokens}, Total Tokens: {usage.total_tokens}")

        df = pd.DataFrame([item.model_dump() for item in response.data.dataset])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"product_listings_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)

        logging.info(f"Data saved to {csv_filename}")

    except UnexpectedModelBehavior as e:
        logging.error(f"Scraping error: {e}")

if __name__ == "__main__":
    main()
