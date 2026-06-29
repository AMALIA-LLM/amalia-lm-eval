#!/bin/bash

# Get current user's running jobs
user=$(whoami)
jobs=$(squeue -u "$user" -h -o "%A")

# Check if there are any jobs
if [ -z "$jobs" ]; then
    echo "No jobs found for user $user"
    exit 0
fi

# Display jobs
echo "Jobs for user $user:"
squeue -u "$user"

# Ask for confirmation
read -p "Do you want to cancel all these jobs? (y/N): " confirmation

if [[ "$confirmation" =~ ^[Yy]$ ]]; then
    echo "Cancelling jobs..."
    scancel -u "$user"
    echo "All jobs cancelled."
else
    echo "Cancelled operation. No jobs were cancelled."
fi