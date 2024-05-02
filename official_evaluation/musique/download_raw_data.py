import argparse
import gdown
import json
import os

def main():

    with open(".all_data_information.json") as file:
        all_data_information = json.load(file)

    for raw_data_filepath, gdrive_id in all_data_information.items():

        if os.path.exists(raw_data_filepath):
            print(f"Not downloading {raw_data_filepath} as it's already available locally.")
            continue

        os.makedirs("raw_data", exist_ok=True)

        print(f"---------------\nDownloading {raw_data_filepath}")
        gdown.download(id=gdrive_id, output=raw_data_filepath, quiet=False)

        if not os.path.exists(raw_data_filepath):
            print(
                f"Looks like the download did not succeed. If the failure was because the drive download quota exceeded, "
                f"please wait or install gdrive (https://github.com/prasmussen/gdrive) and run \n"
                f"gdrive download {gdrive_id} --path .tmp \n"
                f"mv .tmp/{raw_data_filepath.replace('raw_data/', '')} {raw_data_filepath} \n\n"
            )

if __name__ == '__main__':
    main()
