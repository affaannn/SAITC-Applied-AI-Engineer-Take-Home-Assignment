from app.pipeline import RAGPipeline


TEST_QUERIES = [
    "How much notice must an employee give during probation?",
    "What is the current price of Atlas Professional?",
    "What refund window applies to Atlas Enterprise customers?",
]

def main():

    pipeline = RAGPipeline()

    for query in TEST_QUERIES:

        print()
        print("=" * 100)
        print("QUERY:", query)
        print("=" * 100)

        result = pipeline.answer(
            query
        )

        print()
        print("Route:", result["route"])
        print()
        print(result["answer"])

        if "tokens" in result:
            print()
            print("Tokens:", result["tokens"])

        if "documents_used" in result:
            print(
                "Documents:",
                result["documents_used"],
            )

        if "citations" in result:
            print(
                "Citation valid:",
                result[
                    "citations"
                ]["valid"],
            )

if __name__ == "__main__":
    main()