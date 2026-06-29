import ast

import streamlit as st
import pandas as pd
import plotly.express as px
import argparse
import os
import glob
import sys

from model_task_pretty_names import turn_task_to_pretty_name, dataset_to_pretty_name_dict, model_to_pretty_name_dict

# --- Argument Parsing ---
# We must try to parse arguments and catch SystemExit, as Streamlit runs the script
# from top to bottom on every user interaction.

def get_args():
    """Parses command-line arguments for the data folder path."""
    parser = argparse.ArgumentParser(description="Streamlit Visualization for Evaluation Results")
    parser.add_argument(
        "--folder_path",
        type=str,
        # Set a default path if the argument is not provided
        help="Path to the folder containing evaluation results CSV files."
    )

    # We use a try-except block to handle Streamlit's behavior when argparse.parse_args() is called.
    try:
        args = parser.parse_args()
        return args
    except SystemExit:
        # This occurs when Streamlit's execution environment intercepts the help flag.
        # We manually handle the exit to prevent the Streamlit server from quitting.
        # This is a common pattern for Streamlit apps using argparse.
        st.error("Error parsing arguments. Ensure you use '--' to separate Streamlit arguments from script arguments.")
        st.code("streamlit run app.py -- --folder_path /path/to/your/results")
        # Returning None or a default args object can prevent the app from crashing.
        return parser.parse_args([])  # Return default args if SystemExit occurs


args = get_args()
DATA_FOLDER_PATH = args.folder_path

# -----------------------------
# Set wide mode by default
# -----------------------------
st.set_page_config(page_title="Main Metrics Dashboard", layout="wide")

# -----------------------------
# Data Loading and Selector
# -----------------------------

st.sidebar.title("Data Selection")


# Use st.cache_data to speed up loading if the file content doesn't change
@st.cache_data
def shorten(model):
    """Shortens the model name to the last two parts of the path."""
    parts = model.split("/")

    # aim is to grab provider and model name / checkpoint name
    new_model_short = "/".join(parts[-2:]) if len(parts) >= 2 else model

    # instead grab the part before snapshot
    # because of names like this "/models--utter-project--EuroLLM-9B-Instruct/snapshots/f7ae2bc3bcbb538c0b93fa6cfbf388d9898b1ace"
    if "snapshots" in new_model_short:
        new_model_short = parts[-3].replace("models--", "").replace("--", "/")

    return new_model_short


def safe_literal_eval(x):
    """Safely evaluates a string as a Python literal (e.g., list or dict)."""
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    except (ValueError, SyntaxError):
        return [] # Return an empty list if evaluation fails


def load_data(path):
    """Loads and preprocesses the DataFrame."""
    data = pd.read_csv(path)
    # Apply mandatory filtering and shortening after load
    data = data[data["is_main_metric"] == True].copy()
    data["model_short"] = data["model_name"].apply(shorten)
    return data


# --- Folder Validation ---
# st.sidebar.markdown(f"**Data Folder:** `{DATA_FOLDER_PATH}`")
if not os.path.isdir(DATA_FOLDER_PATH):
    st.error(f"Data folder not found or invalid: **{DATA_FOLDER_PATH}**")
    st.markdown("Please start the app with the correct folder path, e.g.:")
    st.code(f"streamlit run {sys.argv[0]} -- --folder_path /path/to/your/results/folder")
    st.stop()

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.csv"))

if not csv_files:
    st.error(f"No .csv files found in the folder: **{DATA_FOLDER_PATH}**")
    st.stop()

# Extract just the filenames for the selector
csv_filenames = [os.path.basename(f) for f in sorted(csv_files)]

# Selector for the CSV file
selected_filename = st.sidebar.selectbox(
    "Select Results File",
    options=csv_filenames,
    help=f"CSV files found in the folder specified by '--folder_path'."
)

selected_csv_path = os.path.join(DATA_FOLDER_PATH, selected_filename)

# Load CSV
try:
    df = load_data(selected_csv_path)
    st.info(f"Loaded: **{selected_filename}** with {len(df)} rows.")

except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# ---------------------------------
# SIDEBAR FILTERS (Same as original)
# ---------------------------------
st.sidebar.title("Filters")

# ---- TASKS: only tasks where NOT subcategories ----
main_task_values = sorted(df[df["is_subcategory"] == False]["main_task"].unique())
selected_tasks = st.sidebar.multiselect(
    "Select Tasks (main tasks only)",
    options=main_task_values,
    default=main_task_values,
    help="Select one or more main tasks. Scrollable if too many tasks."
)

# ---- PT DATASET FILTER ----
pt_dataset_option = st.sidebar.selectbox(
    "Datasets",
    ["All", "PT datasets only", "Other"],
    index=0
)


# ---- DEV/TASK FILTER ----
dev_task_option = st.sidebar.selectbox(
    "Task Type",
    ["All", "Dev", "Test"],
    index=0
)

# ---- SUBCATEGORY toggle ----
include_subcategories = st.sidebar.checkbox("Dataset Subcategories", value=False)

# ---- MULTI-SELECT MODEL ----
model_values = sorted(df["model_short"].unique())
selected_models = st.sidebar.multiselect(
    "Select Models",
    options=model_values,
    default=model_values,
    help="Select one or more models. Scrollable if too many models."
)

# ---- AMALIA MODELS ONLY toggle ----
amalia_only = st.sidebar.checkbox("Only AMALIA models", value=False)

# ---- OPEN SOURCE FILTER ----
open_source = st.sidebar.selectbox(
    "Model Type",
    ["All", "Open Source", "Open Weights"],
)

# ---- TAGS FILTER ----
# 1. Gather all unique tags across all rows
all_tags = set()
if 'tags' in df.columns:
    for tags_list in df['tags'].dropna():
        tags = safe_literal_eval(tags_list)
        if isinstance(tags, list):
            all_tags.update(tags)
    all_tags = sorted(all_tags)

selected_tags = []
if all_tags:
    selected_tags = st.sidebar.multiselect(
        "Filter by Tags (OR logic)",
        options=all_tags,
        default=[],
        help="Select one or more tags. Rows must contain AT LEAST ONE of the selected tags."
    )


# ---------------------------------
# FILTERING DATA
# ---------------------------------
filtered = df.copy()

if selected_tasks:
    filtered = filtered[filtered["main_task"].isin(selected_tasks)]

if not include_subcategories:
    filtered = filtered[filtered["is_subcategory"] == False]

if amalia_only:
    filtered = filtered[filtered["is_amalia_model"] == True]

if selected_models:
    filtered = filtered[filtered["model_short"].isin(selected_models)]

if open_source != "All":
    if open_source == "Open Source":
        open_source_bool = True
    else:
        open_source_bool = False  # Open Weights
    filtered = filtered[filtered["is_open_source_model"] == open_source_bool]

if pt_dataset_option == "PT datasets only":
    filtered = filtered[filtered["is_pt_dataset"] == True]
elif pt_dataset_option == "Other":
    filtered = filtered[filtered["is_pt_dataset"] == False]

if dev_task_option != "All":
    if dev_task_option == "Dev":
        filtered = filtered[filtered["is_dev_task"] == True]
    else:  # Test
        filtered = filtered[filtered["is_dev_task"] == False]


# TAGS FILTERING LOGIC (Using OR logic)
if selected_tags and 'tags' in filtered.columns:
    # The condition is: for each row's 'tags' list, AT LEAST ONE selected tag must be present.
    # We use a lambda function and the 'any()' operator for this "OR" logic.
    filtered = filtered[
        filtered["tags"].apply(lambda row_tags: any(tag in row_tags for tag in selected_tags))
    ]


# ---------------------------------
# PLOT
# ---------------------------------
st.header("Main Metric Values per Task and Model")

if filtered.empty:
    st.warning("No data after filtering!")
else:
    num_tasks = filtered["task"].nunique()
    num_models = filtered["model_short"].nunique()

    # Dynamically increase height
    height = max(600, num_tasks * num_models * 20)

    fig = px.bar(
        filtered,
        x="value",
        y="task_complete",
        color="model_short",
        orientation="h",
        hover_data=["metric", "model_name", "value"],
        text="value",
        text_auto=True
    )

    # Make bars thicker by reducing gaps
    fig.update_layout(
        barmode="group",
        bargap=0.05,  # gap between groups of tasks
        bargroupgap=0.02,  # gap between bars in a group
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=80, r=80, t=20, b=50),
        xaxis_title="Metric Value",
        yaxis_title="Task",
        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14))
    )

    # Increase size of the numbers inside bars
    fig.update_traces(textfont_size=14)

    # Order first by task
    task_order = sorted(filtered["task_complete"].unique(), reverse=True)
    fig.update_yaxes(categoryorder="array", categoryarray=task_order)

    st.plotly_chart(fig, width="content")

# ---------------------------------
# SUMMARY TABLE BELOW THE CHART
# ---------------------------------

if not filtered.empty:
    st.header("Summary per Model")

    # Average performance per model
    avg_perf = filtered.groupby("model_short")["value"].mean().reset_index()
    avg_perf.rename(columns={"value": "avg_performance"}, inplace=True)

    # Compute ranks per task
    filtered["rank"] = filtered.groupby("task")["value"].rank(ascending=False, method="average")
    avg_rank = filtered.groupby("model_short")["rank"].mean().reset_index()
    avg_rank.rename(columns={"rank": "avg_rank"}, inplace=True)

    # Merge summary table
    summary = pd.merge(avg_perf, avg_rank, on="model_short")
    summary = summary.sort_values(by="avg_performance", ascending=False)

    st.dataframe(summary.style.format({"avg_performance": "{:.3f}", "avg_rank": "{:.2f}"}))


# ---------------------------------
# PIVOT TABLE: Models x Tasks
# ---------------------------------

if not filtered.empty:
    st.header("Pivot Table: Models x Tasks")

    # Map pretty names for tasks/datasets
    filtered["task_pretty"] = filtered["task"].map(lambda x: turn_task_to_pretty_name(x.replace("amalia_", "")))

    # Apply pretty names to models
    if model_to_pretty_name_dict:
        filtered["model_pretty"] = filtered["model_name"].map(lambda x: model_to_pretty_name_dict.get(x, x))
    else:
        filtered["model_pretty"] = filtered["model_name"]

    # Use the last part of the model name for better visualization
    filtered["model_pretty"] = filtered["model_pretty"].apply(shorten)

    # Create pivot table
    pivot = filtered.pivot_table(
        index="model_pretty",
        columns="task_pretty",
        values="value",
        aggfunc="first"
    ).fillna("-")

    # Reorder columns according to a predefined order
    if dataset_to_pretty_name_dict:
        desired_col_order = [dataset_to_pretty_name_dict[k] for k in dataset_to_pretty_name_dict if
                             dataset_to_pretty_name_dict[k] in pivot.columns]
        extra_cols = [c for c in pivot.columns if c not in desired_col_order]
        pivot = pivot.reindex(columns=desired_col_order + extra_cols)

    # Reorder rows according to a predefined order
    if model_to_pretty_name_dict:
        desired_row_order = [shorten(model_to_pretty_name_dict.get(m, m)) for m in model_to_pretty_name_dict if
                             shorten(model_to_pretty_name_dict.get(m, m)) in pivot.index]
        extra_rows = [r for r in pivot.index if r not in desired_row_order]
        pivot = pivot.reindex(index=desired_row_order + extra_rows)

        # remove duplicates while preserving order
        pivot = pivot[~pivot.index.duplicated(keep='first')]

    st.dataframe(pivot)


# ---------------------------------
# FULL TABLE VIEW
# ---------------------------------
if not filtered.empty:

    st.header("Full Table View (Filtered Data)")

    st.dataframe(filtered.reset_index(drop=True))



# ---------------------------------
# Warning when some evaluations are missing for some models
# ---------------------------------

if not filtered.empty:
    st.header("Evaluation Completeness Check")

    tasks = filtered["task_complete"].unique()
    models = filtered["model_short"].unique()

    missing_evaluations = []

    for task in tasks:
        for model in models:
            if not ((filtered["task_complete"] == task) & (filtered["model_short"] == model)).any():
                missing_evaluations.append((task, model))

    if missing_evaluations:
        st.warning(f"Incomplete Evaluations. Total: {len(missing_evaluations)}")
        for task, model in missing_evaluations:
            st.write(f"- Task: **{task}**, Model: **{model}**")
    else:
        st.success("All tasks have been evaluated for all selected models!")


# example to run:

# if in remote server before running create the port forwarding (in your local machine) with:
# ssh -L 8501:localhost:8501 user@remote_server

# then run the streamlit app with (in remote server):
# streamlit run streamlit_viz.py --server.port 8501 --server.address 0.0.0.0

# streamlit run --server.port 8501 --server.address 0.0.0.0 streamlit_viz.py -- --folder_path "/path_to_folder_with_csv_files"
