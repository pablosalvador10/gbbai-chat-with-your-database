import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import RawVectorQuery
from dotenv import load_dotenv

from src.aoai.azure_openai import AzureOpenAIManager
from src.database.azuresql import AzureSQLManager
from utils.ml_logging import get_logger


def get_user_query():
    return input(
        "\nPlease enter your database query in natural language, or type 'exit' to quit the program: "
    )


def get_run_query_decision():
    return (
        input("\nDo you want to execute this SQL query? Please enter 'yes' or 'no': ")
        .strip()
        .lower()
    )


def pretty_print_results(results):
    print(
        "\nSQL query executed successfully. The results from the database are as follows:"
    )
    for result in results:
        print(f"- {result[0]}")


def main():
    load_dotenv()

    logger = get_logger()

    DATABASE = "dev-sql-server"
    az_sql_client = AzureSQLManager(database=DATABASE)
    az_client = AzureOpenAIManager()

    service_endpoint = os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    credential = AzureKeyCredential(key)
    index_name = "query-dev-index"

    search_client = SearchClient(service_endpoint, index_name, credential)

    system_message = (
        "You are an expert in databases querying and are assisting a Data Engineer."
    )
    conversation_history = [{"role": "system", "content": system_message}]

    while True:
        user_query = get_user_query()

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting the program. Goodbye!")
            break

        search_vector = az_client.generate_embedding(user_query)
        # hybrid retrieval + rerank
        r = search_client.search(
            user_query,
            top=1,
            vector_queries=[
                RawVectorQuery(vector=search_vector, k=50, fields="table_vector")
            ],
            query_type="semantic",
            semantic_configuration_name="query-index-semantic-config",
            query_language="en-us",
        )

        table_name = None
        for doc in r:
            table_name = doc["table_name"]
            score = doc["@search.score"]
            print(f"\nIdentified Table Name: {table_name}")
            print(f"Relevance Score: {score}")

        if table_name:
            schema = az_sql_client.process_schema(table_name)
            prompt = f"""Based on the database schema below, answer the question. Provide only the answer, avoiding any 
            conversational fluff.\n\nSchema: {schema}\n\n---\n\nQuestion: {user_query}\nAnswer:"""
            content = az_client.generate_chat_response(
                conversation_history=conversation_history,
                query=prompt,
                system_message_content=system_message,
                max_tokens=1000,
            )
            print(f"\nGenerated SQL Query: \n{content}")
            conversation_history.append({"role": "user", "content": user_query})
            conversation_history.append({"role": "assistant", "content": content})
        else:
            logger.info("No table found for the query. Please verify the query.")

        run_query_decision = get_run_query_decision()
        if run_query_decision == "yes":
            try:
                output = az_sql_client.execute_and_fetch(
                    content
                )  # Ensure this method is correctly implemented
                pretty_print_results(output)
            except Exception as e:
                logger.error(
                    f"\nThere seems to be an issue with this query. Please verify the query.\n{e}"
                )


if __name__ == "__main__":
    main()
