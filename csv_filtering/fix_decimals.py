import re

with open('Calc.csv', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace ""number,number"" with "number.number" (convert comma decimals to periods)
# This pattern matches: "" followed by digits, comma, digits, followed by ""
content = re.sub(r'""(\d+),(\d+)""', r'"\1.\2"', content)

with open('Calc.csv', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully converted comma decimals to periods in double-quoted values')
