def generate_prompt_function_correctness(prefix, suffix, generated_code):
    context_gen = f'{prefix[-250:].lstrip()}{generated_code}{suffix[:250].rstrip()}'
    prompt = f'''
Decide whether generated part of code is functionally correct (will it compile) within the nearest context.
For example: Question for each code section: will the generated part compile within this context?
1. Code:
def get_squared_sum(numbers):
    # Return the sum of squares of all numbers in the list
    return sum(x ** 2 for x in numbers)
Generated part: sum(x ** 2 for x in numbers)
Answer: 1

2. Code:
def write_to_file(file_name, content):
    # Write the content to the specified file
    with open(file_name, 'w') as f:
        f.write(content)
Generated part: open(file_name, 'w')
Answer: 0

3. Code:
def find_max_value(values):
    # Find the maximum value in the list of integers
    max_value = values.max()
    return max_value
Generated part: max_value = values.max()
Answer: 0

Apply your knowledge of Python to solve this task.
Here is the code for which need to evaluate:
Code: {context_gen}
Generated part: {generated_code}

Question: Decide whether generated part of code is functionally correct (will it compile) within the nearest context?
Answer format: bool value: 1 or 0
Give your answer here, write only 1 or 0: 
'''
    return prompt