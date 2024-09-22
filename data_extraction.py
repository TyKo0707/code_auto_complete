import re
import pandas as pd


def output_extracted_dataset(data):
    # Print each sample of the dataset with distinguishing important parts by color (cyan in code - middle part)
    color = lambda s: f"\033[96m{s}\033[00m"
    for i in range(len(data)):
        sample = data[i]
        print(f'Index: {color(i + 1)}\n'
              f'Code: {sample["prefix"][-250:].lstrip()}{color(sample["content"])}{sample["suffix"][:250].rstrip()}\n'
              f'Tag: {color(sample["tag"])}\n')


def extract_data_from_match(match, text):
    # Remove all tags before current match
    start_pos = match.regs[-1][0]
    before_match, after_match = text[:start_pos], text[start_pos:]
    text = re.sub(r'\$\$\w+\s*|\$\$', '', before_match) + after_match

    prefix, tag, content, suffix = match.groups()

    # Remove all tags from prefix and suffix
    cleaned_prefix = re.sub(r'\$\$\w+\s*|\$\$', '', prefix)
    cleaned_suffix = re.sub(r'\$\$\w+\s*|\$\$', '', suffix)
    return cleaned_prefix, cleaned_suffix, content, tag, text


def extraction(files_path):
    files_paths_list = [files_path + f for f in os.listdir(files_path)]
    extracted_data = []
    for path in files_paths_list:
        with open(f'{path}', 'r') as file:
            text = file.read() + '\n'

        # Pattern to match the $$tag ... $$ structure
        pattern = re.compile(r'(.*?)\$\$(\w+)\s(.*?)\$\$(.*)', re.DOTALL)
        while match := pattern.match(text):
            cleaned_prefix, cleaned_suffix, content, tag, new_text = extract_data_from_match(match, text)
            text = new_text
            extracted_data.append({'prefix': cleaned_prefix,
                                   'tag': tag,
                                   'content': content,
                                   'suffix': cleaned_suffix,
                                   'file_name': path.split('/')[-1]})
    return extracted_data


if __name__ == '__main__':
    import os
    from environs import Env

    env = Env()
    env.read_env()

    # Extract Python data
    PYTHON_FILES = env.str("PYTHON_FILES")
    PYTHON_SAVE_PATH = env.str("PYTHON_SAVE_PATH")

    extracted_data_py = extraction(PYTHON_FILES)
    df_py = pd.DataFrame(extracted_data_py)
    df_py.to_csv(PYTHON_SAVE_PATH)

    print(f'{"=" * 50} PYTHON DATASET {"=" * 50}\nTags distribution: {df_py.tag.value_counts()}\n')
    output_extracted_dataset(extracted_data_py)

    # Extract C/C# data
    C_FILES = env.str("C_FILES")
    C_SAVE_PATH = env.str("C_SAVE_PATH")

    extracted_data_c = extraction(C_FILES)
    df_c = pd.DataFrame(extracted_data_c)
    df_c.to_csv(C_SAVE_PATH)

    print(f'{"=" * 50} C/C# DATASET {"=" * 50}\nTags distribution: {df_c.tag.value_counts()}\n')
    output_extracted_dataset(extracted_data_c)
