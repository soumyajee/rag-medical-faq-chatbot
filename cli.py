import pandas as pd
from rag_chatbot import rag_query   # ✅ this is a function, not a class
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_cli():
    # Specific queries from assignment
    specific_queries = [
        "What are the early symptoms of diabetes?",
        "Can children take paracetamol?",
        "What foods are good for heart health?"
    ]
    results = []

    # Process specific queries
    print("\nProcessing required queries...")
    logger.info("Processing specific queries...")
    for query in specific_queries:
        try:
            print(f"\nQuery: {query}")
            answer = rag_query(query)   # ✅ directly call the function
            print(f"Answer: {answer}")
            results.append({"Question": query, "Answer": answer})
            logger.info(f"Processed query '{query}': {answer[:100]}...")
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            print(f"Error processing query '{query}': {e}")
            results.append({"Question": query, "Answer": f"Error: {e}"})

    # Interactive CLI loop
    print("\nMedical FAQ Chatbot (type 'exit' to quit, 'save' to save results to CSV)")
    while True:
        query = input("\nEnter your medical question: ").strip()
        if query.lower() == 'exit':
            break
        if query.lower() == 'save':
            try:
                if results:
                    df = pd.DataFrame(results)
                    df.to_csv('query_results.csv', index=False)
                    print("Results saved to query_results.csv")
                    logger.info("Results saved to query_results.csv")
                else:
                    print("No results to save.")
                    logger.warning("No results to save to CSV")
            except Exception as e:
                logger.error(f"Error saving to CSV: {e}")
                print(f"Error saving to CSV: {e}")
            continue
        if query:
            try:
                print(f"\nQuery: {query}")
                answer = rag_query(query)   # ✅ directly call the function
                print(f"Answer: {answer}")
                results.append({"Question": query, "Answer": answer})
                logger.info(f"Processed query '{query}': {answer[:100]}...")
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                print(f"Error: {e}")
                results.append({"Question": query, "Answer": f"Error: {e}"})

    # Save results on exit
    try:
        if results:
            df = pd.DataFrame(results)
            df.to_csv('query_results.csv', index=False)
            print("Results saved to query_results.csv")
            logger.info("Results saved to query_results.csv")
        else:
            logger.warning("No results to save on exit")
            print("No results to save.")
    except Exception as e:
        logger.error(f"Error saving to CSV on exit: {e}")
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    run_cli()
