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

    parser = argparse.ArgumentParser(description='Download models.')
    parser.add_argument(
        'experiment_name',
        type=str,
        choices=experiment_names+["all"],
        help='name of the experiment (model) for which to you want to download model. '
             'Use "all" to download all models.'
    )
    args = parser.parse_args()

    for dataset_name, experiment_details in all_experiment_information.items():

        for experiment_detail in experiment_details:
            for experiment in experiment_detail["experiments"]:
                experiment_name = experiment["experiment_name"]
                if experiment_name != args.experiment_name and args.experiment_name != "all":
                    continue

                gdrive_id = experiment["gdrive_id"]
                output_directory = os.path.join("serialization_dir", experiment_name)
                output_filepath = os.path.join(output_directory, "model.tar.gz")
                if os.path.exists(output_filepath):
                    print(f"Not downloading {output_filepath} as it's already available locally.")
                    continue

                os.makedirs(output_directory, exist_ok=True)

                print(f"---------------\nDownloading and saving model at: {output_filepath}")
                gdown.download(id=gdrive_id, output=output_filepath, quiet=False)

                if not os.path.exists(output_filepath):
                    print(
                        f"Looks like the download did not succeed. If the failure was because the drive download quota exceeded, "
                        f"please wait or install gdrive (https://github.com/prasmussen/gdrive) and run \n"
                        f"gdrive download {gdrive_id} --path .tmp \n"
                        f"mv .tmp/{experiment_name}.tar.gz {output_filepath} \n\n"
                    )

if __name__ == '__main__':
    main()
