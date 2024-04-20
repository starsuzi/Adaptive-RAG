import os
import requests
import argparse
import subprocess
import _jsonnet
import json


def is_elasticsearch_running() -> bool:
    try:
        res = requests.get("http://localhost:9200/_cluster/health")
        if res.status_code == 200:
            if res.json()["number_of_nodes"] > 0:
                return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Start/stop or check status of elasticsearch server.")
    parser.add_argument("command", type=str, help="start, stop or check status", choices=("start", "stop", "status"))
    args = parser.parse_args()

    es_pid_path = os.path.expanduser("~/.es_pid")
    elasticsearch_path = json.loads(_jsonnet.evaluate_file(".global_config.jsonnet"))["ELASTICSEARCH_PATH"]

    if args.command == "start":

        if os.path.exists(es_pid_path):
            exit("ES PID file aleady exists. Turn off ES first.")

        command = 'ES_JAVA_OPTS="-Xms26g -Xmx26g" '  # larger heapsize needed for natcq
        command += f"{elasticsearch_path} --daemonize --silent --pidfile {es_pid_path}"
        subprocess.call(command, shell=True)

    elif args.command == "stop":

        if not os.path.exists(es_pid_path):
            exit("ES PID file not found. Recheck if it's running or turn it off manually.")

        command = f"pkill -F {es_pid_path}"
        subprocess.call(command, shell=True)

        if os.path.exists(es_pid_path):
            os.remove(es_pid_path)

    elif args.command == "status":

        if is_elasticsearch_running():
            print("Elasticsearch is running.")
        else:
            print("Elasticsearch is NOT running.")

        if os.path.exists(es_pid_path):
            print("ES PID file does exist.")
        else:
            print("ES PID file does NOT exist.")


if __name__ == "__main__":
    main()
