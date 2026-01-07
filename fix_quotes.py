with open('Calc.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    # Remove the newline to check the line content
    line_content = line.rstrip('\n')
    
    # If line starts and ends with a single quote, remove it
    if line_content.startswith('"') and line_content.endswith('"'):
        line_content = line_content[1:-1]
    
    # Add the newline back
    fixed_lines.append(line_content + '\n')

with open('Calc.csv', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print('Successfully removed 1 quote from beginning and end of each row')
