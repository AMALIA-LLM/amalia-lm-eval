from typing import List
import re

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from model_task_pretty_names import turn_model_to_pretty_name


def plot_graphs_for_dfs(df_to_plot: pd.DataFrame, output_folder: str = None, file_name: str = None):
    """
    Plots a grouped bar chart where each bar is a model, tasks are on the y-axis,
    and the x-axis is the metric value.

    The DataFrame must have the columns:
    - 'task_complete' (str): The task name (y-axis categories).
    - 'model_name' (str): The model name (grouping/color variable).
    - 'value' (numeric): The metric value (x-axis bar length).
    """
    # Sort model names with proper checkpoint ordering
    def get_sort_key(model_name: str):
        match = re.search(r'checkpoint-(\d+)', model_name)
        if match:
            # Return tuple with negative checkpoint number for descending order
            checkpoint_num = -int(match.group(1))  # Negative for reverse sort
            base_name = model_name[:match.start()]
            return (base_name, checkpoint_num)
        # Models without checkpoint go last when reversed
        return (model_name, float('inf'))

    # Create categorical type with sorted order
    sorted_models = sorted(df_to_plot['model_name'].unique(), key=get_sort_key)
    df_to_plot = df_to_plot.copy()
    df_to_plot['model_name'] = pd.Categorical(df_to_plot['model_name'], categories=sorted_models, ordered=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(20, 30))

    barplot = sns.barplot(
        data=df_to_plot,
        x='value',
        y='task_complete',
        hue='model_name',
        errorbar=None,  # Remove error bars
        palette="Spectral"
    )

    for container in barplot.containers:
        # container.patches holds all the Rectangles (bars) for one model group
        for patch in container.patches:

            # Format the text (e.g., to 2 decimal places)
            text = f'{patch.get_width():.2f}'

            # Calculate adaptive font size based on number of models
            num_models = len(sorted_models)
            if num_models <= 3:
                fontsize = 9
            elif num_models <= 5:
                fontsize = 7
            elif num_models <= 8:
                fontsize = 6
            else:
                fontsize = max(4, 9 - num_models // 2)  # Minimum font size of 4

            # Use plt.text() to place the label
            plt.text(
                x=patch.get_width() + 0.01,  # x-position (value + a small offset)
                y=patch.get_y() + patch.get_height() / 2,  # y-position (center of the bar)
                s=text,  # The formatted text
                ha='left',  # Horizontal alignment: left (starts text after the bar)
                va='center',  # Vertical alignment: center (centers text vertically)
                fontsize=fontsize  # Adjust as needed
            )

    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Task')

    # Rotate y-axis labels if needed
    plt.yticks(rotation=0)

    # Calculate average and median for each model
    legend_labels = []
    for model in sorted_models:
        model_data = df_to_plot[df_to_plot['model_name'] == model]['value']
        avg = model_data.mean()
        median = model_data.median()
        legend_labels.append(f'{model} (μ={avg:.2f}, M={median:.2f})')

    # Get handles from the barplot
    handles, _ = barplot.get_legend_handles_labels()

    # Move legend outside the plot with calculated statistics
    plt.legend(
        handles=handles,
        labels=legend_labels,
        title='Model (μ=Average, M=Median)',
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        fontsize=8
    )

    # Adjust layout to fit everything
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for legend

    if output_folder and file_name:
        output_path = f"{output_folder}/{file_name}_graph.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  # bbox_inches='tight' ensures legend is included
        print(f"Graph saved to {output_path}")

    # Display the plot
    # plt.show()


def create_graph_per_model_task(
        csv_results_path: str,
        considered_tasks: List[str] = None,
        considered_models: List[str] = None,
        exclude_models: List[str] = None,
        output_folder: str = None,
        min_checkpoint: int = None, max_checkpoint: int = None, checkpoints: str = None):

    # read the csv
    df = pd.read_csv(csv_results_path, header=0)

    # task column
    if "task_complete" in df.columns:
        task_column = "task_complete"
    else:
        task_column = "task"

    def rename_amalia(col_name):
        if col_name.strip().startswith("amalia_"): #and ("mt-pt" in col_name or "mt_pt" in col_name):
            return col_name.replace("amalia_", "")
        return col_name

    # rename task_column values with amalia prefix just to make them be ordered correctly
    df[task_column] = df[task_column].apply(rename_amalia)

    # filter the tasks and models
    if considered_tasks:
        # keep only the considered tasks in "task" column
        df = df[df[task_column].isin(considered_tasks)]
    else:
        # reverse sort the tasks alphabetically
        df = df.sort_values(task_column, ascending=False)

    # filter the models
    if considered_models:
        # keep only the considered models in "model_name" column
        df = df[df["model_name"].isin(considered_models)]

    if exclude_models:
        # exclude the models in "model_name" column
        df = df[~df["model_name"].isin(exclude_models)]

    # Sort columns to ensure the task column is first and checkpoints are sorted
    checkpoint_rows = df["model_name"]

    # Filter checkpoints based on parameters
    if checkpoints is not None or min_checkpoint is not None or max_checkpoint is not None:
        filtered_rows = []

        # Parse checkpoints parameter if provided
        selected_checkpoints = None
        if checkpoints is not None:
            selected_checkpoints = set(int(x.strip()) for x in checkpoints.split(','))

        for row in checkpoint_rows:
            match = re.search(r'checkpoint-(\d+)', row)
            if match:
                ckpt_num = int(match.group(1))
                # Include checkpoint if it matches the criteria
                if selected_checkpoints is not None:
                    # Use specific checkpoints list
                    if ckpt_num in selected_checkpoints:
                        filtered_rows.append(row)
                else:
                    # Use minimum and/or maximum checkpoint threshold
                    include = True
                    if min_checkpoint is not None and ckpt_num < min_checkpoint:
                        include = False
                    if max_checkpoint is not None and ckpt_num > max_checkpoint:
                        include = False
                    if include:
                        filtered_rows.append(row)
            else:
                # Keep columns that don't have checkpoint numbers
                filtered_rows.append(row)

        df = df[df["model_name"].isin(filtered_rows)]

    df.set_index(task_column, inplace=True)

    df['model_name'] = df['model_name'].apply(turn_model_to_pretty_name)

    # order by task name and then by is_main_metric and is_subcategory
    df = df.sort_index()
    df = df.sort_values(by=['is_main_metric', 'is_subcategory'], ascending=[False, True])

    # now we will create different dfs for various datasets

    # now one for the main with everything - is_subcategory = False, is_main_metric = True
    df_main = df[(df['is_subcategory'] == False) & (df['is_main_metric'] == True)]

    # plot the main df
    plot_graphs_for_dfs(df_main, output_folder=output_folder, file_name="tasks_main")

    # get unique tasks
    unique_tasks = df_main.index.unique().tolist()
    print("Unique tasks:", len(unique_tasks), unique_tasks)

    # get unique models
    unique_models = df_main['model_name'].unique().tolist()
    print("Unique models:", len(unique_models), unique_models)

    # it is pt if it mentions the word amalia in task or has mt-pt or mt_pt
    def is_pt_dataset(task_name: str) -> bool:
        task_name = task_name.lower()
        if "amalia" in task_name or "mt-pt" in task_name or "mt_pt" in task_name or "pt_" in task_name or "_port" in task_name or "_pt" in task_name or "alba" in task_name:
            return True
        return False

    # one for the pt datasets only and one for the en datasets only
    df_pt = df_main[df_main.index.to_series().apply(is_pt_dataset)]
    if not df_pt.empty:
        plot_graphs_for_dfs(df_pt, output_folder=output_folder, file_name="tasks_pt")

    df_en = df_main[~df_main.index.to_series().apply(is_pt_dataset)]
    if not df_en.empty:
        plot_graphs_for_dfs(df_en, output_folder=output_folder, file_name="tasks_en")

    # one for each dataset with subcategories where the first row is the main results and then is the subcategories
    # i.e. is_main_metric = True, first row of df should be the main results for that task and then the subcategories
    df_with_subcategories = df[df['is_main_metric'] == True]

    # get main task names with subcategories
    tasks_with_subcategories = df_with_subcategories[df_with_subcategories['is_subcategory'] == True]["main_task"]
    # get unique main task names
    tasks_with_subcategories = tasks_with_subcategories.unique().tolist()

    # now for each task with subcategories, we will create a df and plot it where the first row is the main results and then the subcategories
    for task in tasks_with_subcategories:
        df_task = df_with_subcategories[df_with_subcategories['main_task'] == task]
        if not df_task.empty:
            # create a safe file name
            safe_task_name = task.replace(" ", "_").replace("/", "_").replace("\\", "_")
            plot_graphs_for_dfs(df_task, output_folder=output_folder, file_name=f"task_{safe_task_name}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder with results.csv")
    parser.add_argument("--include_models", type=str, nargs="+", default=None, help="Choose one of the predefined model sets")
    parser.add_argument("--exclude_models", type=str, nargs="*", default=None,)
    parser.add_argument("--min_checkpoint", type=int, default=None,
                        help="Minimum checkpoint number to include (e.g., 500 to include only checkpoint-500 and above)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint numbers to include (e.g., '100,300,500')")
    parser.add_argument("--max_checkpoint", type=int, default=None,
                        help="Maximum checkpoint number to include (e.g., 500 to include only up to checkpoint-500)")
    args = parser.parse_args()

    create_graph_per_model_task(
        f"{args.folder_path}/results.csv",
        considered_tasks=[],
        considered_models=args.include_models,
        exclude_models=args.exclude_models,
        output_folder=args.folder_path,
        min_checkpoint=args.min_checkpoint,
        max_checkpoint=args.max_checkpoint,
        checkpoints=args.checkpoints
    )


if __name__ == "__main__":
    main()
