import json
import csv
import os

def safe_get_first(metrics, key):
    """
    Safely returns the first element of the list in metrics[key],
    or None if the key is missing or the list is empty.
    """
    lst = metrics.get(key, None)
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    else:
        return None

def json_to_csv(json_path, output_dir, output_filename="metrics.csv"):
    """
    Converts the JSON file with metrics into a CSV file.

    Args:
        json_path (str): Full path to the .json file.
        output_dir (str): Directory where the .csv file will be saved.
        output_filename (str): Name of the CSV file (optional).
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for dataset_name, methods in data.items():
        # Only include datasets ending with "_norm"
        if not dataset_name.endswith("_norm"):
            continue
        
        for method_key, metrics in methods.items():
            base_method = ""
            centrality = ""
            proportion = ""

            if method_key.startswith("SSKISOMAP"):
                parts = method_key.split("_")
                base_method = "SSKISOMAP"
                if len(parts) >= 4:
                    centrality = parts[1]
                    proportion = parts[-1].replace("p", "")
            else:
                base_method = method_key

            row = {
                "dataset": dataset_name,
                "method": method_key,
                "base_method": base_method,
                "centrality": centrality,
                "proportion": proportion,
                "ri": safe_get_first(metrics, "ri"),
                "ch": safe_get_first(metrics, "ch"),
                "fm": safe_get_first(metrics, "fm"),
                "v": safe_get_first(metrics, "v"),
                "dbs": safe_get_first(metrics, "dbs"),
                "ss": safe_get_first(metrics, "ss"),
            }
            rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    fieldnames = [
        "dataset", "method", "base_method", "centrality", "proportion",
        "ri", "ch", "fm", "v", "dbs", "ss"
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    #file = 'dataset_results_paper_setup_min_max_v2'
    #file = 'dataset_results_MedMNIST'
    file = 'dataset_results_paper_setup_min_sum'
    json_file = 'results/json/' + file + '.json'
    output_directory = "results/csv"
    json_to_csv(json_file, output_directory, file + '.csv')
