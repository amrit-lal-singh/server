import re

def add_double_quotes(match):
    return f'"{match.group(1)}":'

file_path = '/home/amrit/physics/server/data/11.txt'
output_path = '67.json'

with open(file_path, 'r') as input_file:
    input_text = input_file.read()

# Add double quotes around property keys
json_text = re.sub(r'([a-zA-Z0-9_]+)(?=\s*:\s*)', add_double_quotes, input_text)

# Write the modified content to the output file
with open(output_path, 'w') as output_file:
    output_file.write(json_text)

print(f'Double-quoted JSON written to {output_path}')
