def generate_prompt_function_correctness(prefix, suffix, generated_code):
    context_gen = f'{prefix[-250:].lstrip()}{generated_code}{suffix[:250].rstrip()}'
    prompt = f'''
Decide whether generated part of code is functionally correct (will it compile) within the nearest context.
For example: Question for each code section: will the generated part compile within this context?
1. Code:
def generate_time_based_id():
    # Get the current time in the format YYYYMMDDHHMMSSFFF (year, month, day, hour, minute, second, millisecond)
    return datetime.now().strftime("%Y%m%d_%H%M%S")
Generated part: datetime.now().strftime("%Y%m%d_%H%M%S")
Answer: 1

2. Code:
def parse(message):
    # Extracts the substring between <s> and </s> tags in the given message
    message = message.split("<s>")
    return message[start:end]
Generated part: message = message.split("<s>")
Answer: 0

3. Code:
file_path = '/content/drive/MyDrive/CodeCompletion/functions_df_inputs_outputs.csv'
functions_df = pd.load_csv(file_path)
functions_df[columns_to_convert] = functions_df[columns_to_convert].astype(str)
print(f'[functions_df.iloc[0]]')
Generated part: load_csv(file_path)
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