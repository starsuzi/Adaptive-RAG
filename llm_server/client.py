import requests
import json


def main():

    # This is just an example. Change as necessary.
    host = "http://aristo-cirrascale-13.reviz.ai2.in"
    port = 49171

    params = {"prompt": "Hello, I am unconscious and"}  # see other arguments in serve_models/main:generate
    response = requests.get(host + ":" + str(port) + "/generate", params=params)
    result = response.json()

    result.get("message", "")
    result.get("generated_text", "")
    result.get("model_name", "")  # To assure that response is from the right model.

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
