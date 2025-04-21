import os
import json

input_folder = "context_info"
output_file = "sanitized_sanbox_data_trajectory.json"

# list all files in the input folder
context_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

sanitized_contexts = []

# process each file
for i, file_name in enumerate(sorted(context_files), start=1):
    file_path = os.path.join(input_folder, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
        if "trajectory" in data:
            # sanitize trajectory part
            sanitized_context = {
                "main_number": i,
                "trajectory": {k: v for k, v in data["trajectory"].items() if k != "sensitive_info_items"}
            }
            sanitized_contexts.append(sanitized_context)

# write the sanitized contexts to the output file
with open(output_file, 'w') as f:
    json.dump(sanitized_contexts, f, indent=4)

print(f"Sanitized data has been written to {output_file}")