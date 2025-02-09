# import re

# def exract_first_number1(text):
#     return text.split("#")[-1].replace(",", "").strip()

# def extract_first_number2(text):
#     # updated regex to capture the first number after '###'
#     match = re.search(r'###\s*(\d+\.?\d*)', text)
#     if match:
#         return match.group(1)
#     return None


# test_cases = [
#     "### 0.0", 0.0,
#     "### 0.0, 0.0", 0.0,
#     "### 0.0 ### 0.0", 0.0,
#     "### 1110.0### 0.0 0.033", 1110.0,
#     '699 3 ##  45 ###34 ### 35#', 34,
#     '### 15### 16###', 15,
#     '### 40### 41### 42### 43', 40,
# ]

# for i in range(0, len(test_cases), 2):
#     input = test_cases[i]
#     r = str(test_cases[i+1])
#     r2 = extract_first_number2(input)
#     r1 = exract_first_number1(input)
#     print(f"Test {i}: `{input}` -> r1={r1} {r1==r} == r2={r2} {r2==r} == r={r}")
    

import json
f = "./data/gsm_train.json"

with open(f, "r") as f:
    data = json.load(f)
print(len(data))
