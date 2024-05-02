import argparse
import gdown
import json
import os

def main():

    with open(".all_experiment_information.json") as file:
        all_experiment_information = json.load(file)

    experiment_names = [
        experiment["experiment_name"]
        for dataset_name, experiment_details in all_experiment_information.items()
        for experiment_detail in experiment_details
        for experiment in experiment_detail["experiments"]
    ]

    parser = argparse.ArgumentParser(description='Download raw predictions.')
    parser.add_argument(
        'experiment_name',
        type=str,
        choices=experiment_names+["all"],
        help='name of the experiment (model) for which to you want to download predictions. '
             'Use "all" to download all raw predictions.'
    )
    args = parser.parse_args()

    for dataset_name, experiment_details in all_experiment_information.items():
        for experiment_detail in experiment_details:
            for experiment in experiment_detail["experiments"]:

                experiment_name = experiment["experiment_name"]
                if experiment_name != args.experiment_name and args.experiment_name != "all":
                    continue

                predictions = experiment["predictions"]
                for prediction in predictions:
                    prediction_gdrive_id = prediction["gdrive_id"]
                    prediction_filepath = prediction["path"]
                    prediction_filepath = prediction_filepath.replace("raw_data/", "").replace("/", "__")
                    save_path = os.path.join(
                        "serialization_dir", experiment_name, "predictions", prediction_filepath
                    )
                    assert save_path.count("/") == 3
                    if os.path.exists(save_path):
                        print(f"Not downloading {save_path} as it's already available locally.")
                        continue

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    print(f"---------------\nDownloading saving predictions at: {save_path}")
                    gdown.download(id=prediction_gdrive_id, output=save_path, quiet=False)

                    if not os.path.exists(save_path):
                        print(
                            f"Looks like the download did not succeed. If the failure was because the drive download quota exceeded, "
                            f"please wait or install gdrive (https://github.com/prasmussen/gdrive) and run \n"
                            f"gdrive download {prediction_gdrive_id} --path .tmp \n"
                            f"mv .tmp/{'____'.join([experiment_name, prediction_filepath])} {save_path} \n\n"
                        )

if __name__ == '__main__':
    main()
