import requests
import json
import argparse

from pygments import highlight, lexers, formatters


def main():

    parser = argparse.ArgumentParser(description="Query retriever interactively.")
    parser.add_argument(
        "--retrieval_method",
        type=str,
        help="retrieval_method",
        choices={
            "retrieve_from_elasticsearch",
            "retrieve_from_blink",
            "retrieve_from_blink_and_elasticsearch",
            "retrieve_from_dpr",
            "retrieve_from_contriever",
        },
        required=True,
    )
    parser.add_argument("--host", type=str, help="host", required=True)
    parser.add_argument("--port", type=int, help="port", required=True)  # 443 is default for ngrok
    parser.add_argument("--max_hits_count", type=int, help="max_hits_count", default=5, required=False)
    args = parser.parse_args()

    while True:
        query_text = input("Enter Query: ")

        params = {
            # choices: "retrieve_from_elasticsearch", "retrieve_from_blink",
            # "retrieve_from_blink_and_elasticsearch", "retrieve_from_dpr",
            # retrieve_from_contriever
            "retrieval_method": args.retrieval_method,
            ####
            "query_text": query_text,
            "max_hits_count": args.max_hits_count,
        }

        url = args.host.rstrip("/") + ":" + str(args.port) + "/retrieve"
        result = requests.post(url, json=params)

        if result.ok:

            result = result.json()
            retrieval = result["retrieval"]
            time_in_seconds = result["time_in_seconds"]
            retrieval_str = json.dumps(retrieval, indent=4)
            retrieval_str = highlight(retrieval_str.encode("utf-8"), lexers.JsonLexer(), formatters.TerminalFormatter())

            print(f"Time taken in seconds: {time_in_seconds}")
            print(retrieval_str)

        else:
            print("Something went wrong!\n\n")


if __name__ == "__main__":
    main()
