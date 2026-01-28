import re

with open('Calc.csv', 'r', encoding='utf-8') as f:
    content = f.read()

# First, replace all "" with " (convert double quotes to single quotes)
content = content.replace('""', '"')

# Then, convert numeric decimal patterns: "digits,digits" to "digits.digits"
content = re.sub(r'"(\d+),(\d+)"', r'"\1.\2"', content)

with open('Calc.csv', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully fixed all quoted strings and numeric decimals')
