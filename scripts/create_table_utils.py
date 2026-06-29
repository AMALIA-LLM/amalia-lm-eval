import os.path
import pandas as pd
from model_task_pretty_names import turn_task_to_pretty_name, dataset_to_pretty_name_dict, model_to_pretty_name_dict


def create_tables(file_name: str, outfile: str = "results_by_model_ordered.csv", extra_tables: bool = True):

    # Check if file exists before reading
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"The input file was not found: {file_name}")

    df = pd.read_csv(file_name)

    # --- 1. Task Name Prettification ---
    df['task_complete'] = df['task_complete'].apply(turn_task_to_pretty_name, verbose=True)

    # --- 2. Model Name Prettification ---
    # rename the columns based on the model_to_pretty_name_dict
    df = df.rename(columns=model_to_pretty_name_dict)

    # transpose the dataframe
    df = df.set_index('task_complete').transpose()

    # multiply by 100 and round to 1 decimal places
    df = (df * 100).round(1)

    # --- 3. Sort columns (tasks) based on dataset_to_prety_name_dict order ---
    # Get unique pretty names in order from dataset_to_prety_name_dict
    task_order = list(dict.fromkeys(dataset_to_pretty_name_dict.values()))  # Remove duplicates while preserving order
    # Get columns that exist in the dataframe
    existing_tasks = [task for task in task_order if task in df.columns]
    # Get columns not in the dictionary (put at end)
    other_tasks = [task for task in df.columns if task not in task_order]
    # Reorder columns
    df = df[existing_tasks + other_tasks]

    # --- 4. Sort rows (models) based on model_to_pretty_name_dict order ---
    # Get unique model names in order from model_to_pretty_name_dict
    model_order = list(dict.fromkeys(model_to_pretty_name_dict.values()))  # Remove duplicates while preserving order
    # Get rows that exist in the dataframe
    existing_models = [model for model in model_order if model in df.index]
    # Get rows not in the dictionary (put at end)
    other_models = [model for model in df.index if model not in model_order]
    # Reorder rows
    df = df.reindex(existing_models + other_models)


    # --- 5. Calculate Borda Count ---
    # 5a. Determine N (number of candidates/models)
    N = len(df)

    # 5b. Rank the models for each task (column).
    # We only rank on the tasks defined in names_dict (existing_tasks)
    tasks_to_rank = existing_tasks

    # Define safety benchmarks where lower is better
    # TODO needs to be more easily maintainable
    safety_benchmarks = [
        'AdvBench', 'SST', 'XSTest', 'HarmBench-S', 'HarmBench-C', 'WildGuard', 'ToxicChat',
        'AdvBench (en)', 'SST (en)', 'XSTest (en)', 'HarmBench-S (en)', 'HarmBench-C (en)',
        'WildGuard (en)', 'ToxicChat (en)'
    ]

    # Separate tasks into safety and non-safety
    safety_tasks = [task for task in tasks_to_rank if task in safety_benchmarks]
    non_safety_tasks = [task for task in tasks_to_rank if task not in safety_benchmarks]

    # Rank safety benchmarks (lower is better, so ascending=True)
    df_ranks_safety = df[safety_tasks].rank(axis=0, method='min', ascending=True) if safety_tasks else pd.DataFrame(index=df.index)

    # Rank non-safety benchmarks (higher is better, so ascending=False)
    df_ranks_non_safety = df[non_safety_tasks].rank(axis=0, method='min', ascending=False) if non_safety_tasks else pd.DataFrame(index=df.index)

    # Combine the ranks, preserving the original task order
    df_ranks = pd.concat([df_ranks_non_safety, df_ranks_safety], axis=1)[tasks_to_rank]

    # 5c. Calculate Borda Points: Points = N - Rank + 1
    # This assigns N points to rank 1, N-1 to rank 2, ..., 1 point to rank N.
    df_borda_points = N - df_ranks + 1

    # 5d. Sum Borda Points across all ranked tasks for each model (row)
    df['Borda Count'] = df_borda_points.sum(axis=1).round(1)

    # now lets calculate a variation of Borda count which account for the average rank position
    # 5e. Calculate Average Rank Position across all ranked tasks for each model (row)
    df['Average Rank'] = df_ranks.mean(axis=1).round(2)

    # ---- Save as csv
    output_file = os.path.join(os.path.dirname(file_name), outfile)
    df.to_csv(output_file)

    def pt_exams_tables():
        # new lets save one special table for the pt-exams
        pt_exams_tasks = [task for task in df.columns if task.startswith('PT-E') and 'CoT' not in task]
        df_pt_exams = df[pt_exams_tasks]
        if df_pt_exams.empty:
            print("No PT-E tasks found in the dataframe.")
        else:
            output_file_pt_exams = os.path.join(os.path.dirname(file_name), output_file.replace(".csv", "_pt_exams.csv"))
            df_pt_exams.to_csv(output_file_pt_exams)

        # and another for pt-exams-cot
        pt_exams_cot_tasks = [task for task in df.columns if task.startswith('PT-E') and 'CoT' in task]
        df_pt_exams_cot = df[pt_exams_cot_tasks]
        if df_pt_exams_cot.empty:
            print("No PT-E (CoT) tasks found in the dataframe.")
        else:
            output_file_pt_exams_cot = os.path.join(os.path.dirname(file_name), output_file.replace(".csv", "_pt_exams_cot.csv"))
            df_pt_exams_cot.to_csv(output_file_pt_exams_cot)

    if extra_tables:
        pt_exams_tables()

    def alba_tables():
        # and one for alba and its subcategories
        alba_tasks = [task for task in df.columns if task.startswith('ALBA') and not task.startswith('ALBA-e')]
        df_alba = df[alba_tasks]
        if df_alba.empty:
            print("No ALBA tasks found in the dataframe.")
        else:
            output_file_alba = os.path.join(os.path.dirname(file_name), output_file.replace(".csv", "_alba.csv"))
            df_alba.to_csv(output_file_alba)

    if extra_tables:
        alba_tables()

    print(f"Successfully processed data and saved to: {output_file}")
