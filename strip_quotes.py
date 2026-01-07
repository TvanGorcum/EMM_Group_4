with open('data_calculus_new.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    if line.startswith('"') and line.rstrip().endswith('"'):
        line = line[1:]  # Remove first quote
        line = line.rstrip()[:-1] + '\n'  # Remove last quote before newline
    cleaned_lines.append(line)

with open('data_calculus_new.csv', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print('Quotation marks removed from beginning and end of each row')
