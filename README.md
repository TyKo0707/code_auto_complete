# Code completion

The project's focus is on code completion. The pipeline separates code samples into three sections: prefix (before the
cursor), middle (missing code), and suffix (after the cursor).
The middle section's completions are generated using pre-trained open-source
models: `tiny_starcoder_py`, `starcoder2_3b`, `starcoder2_7b`, and `starcoder2_15b`.

After creating completions, the outputs are examined both manually and automatically using the target code.

The goal is to establish which metrics correspond best with human evaluation and to assess the quality of code
completion across various models.

## Executing program

I didn't create a main file for executing the program because this doesn't really make sense, but you can run all the
steps using the following instructions:

0. **Installing requirements and setting env-variables**: Install all the requirements from the `requirements.txt` file.
   You also need to assign environment variables at `.env` in the following way:

```python
DATA_PATH='/path_to_project/data'
PYTHON_FILES='/path_to_project/data/python_lang/'
PYTHON_SAVE_PATH='/path_to_project/data/python_dataset.csv'
C_FILES='/path_to_project/data/c_lang/'
C_SAVE_PATH='/path_to_project/data/c_lang_dataset.csv'
CLAUDE_KEY='your-claude-key'
MODELS_LIST='tiny_starcoder_py,starcoder2_3b,starcoder2_7b,starcoder2_15b'
EMBEDDING_CHECKPOINT='Salesforce/codet5p-110m-embedding'
OPENAI_KEY='your-openai-key'
```

1. **Explore Models**:
   Run `models_explore/explore_big_starcoder.ipynb` and `models_explore/explore_tiny_starcoder.ipynb`.
2. **Data Extraction**:
   Execute `data_extraction.py` (input: `PYTHON_FILES`, output: `PYTHON_SAVE_PATH`).
3. **Generate Completions**:
   Run `generate_completions.ipynb` (input: `PYTHON_SAVE_PATH`, output: `DATA_PATH/python_dataset_gen.csv`).
4. **Evaluate Predictions**:
   Use `manual_evaluation.py` and `automatic_evaluation.py` on `DATA_PATH/python_dataset_gen.csv` (
   output: `DATA_PATH/eval_results/manual` and `DATA_PATH/eval_results/automatic`).
5. **Create Visualizations**:
   Run `create_visualizations.py` (input: `DATA_PATH/eval_results/manual` and `automatic`).

## Steps of solution

### 0. Exploring the models

* Explore different
  models ([explore_tiny_starcoder.ipynb](https://github.com/TyKo0707/code_completion/blob/main/models_explore/explore_tiny_starcoder.ipynb), [explore_big_starcoder.ipynb](https://github.com/TyKo0707/code_completion/blob/main/models_explore/explore_big_starcoder.ipynb))
  specified for code generation to find the most suitable for the task.
* Find the optimal config and method of generation for each model.
* Result of this step: decided to use starcoder and starcoder-based
  models `tiny_starcoder_py`, `starcoder2_3b`, `starcoder2_7b`, and `starcoder2_15b`.

### 1. Data Collection and Preprocessing

I need to create a dataset consists of prefix, middle part (target), and suffix.

* Took a few files ([data/python_lang](https://github.com/TyKo0707/code_completion/tree/main/data/python_lang)) from the
  local repository and annotated them.
  Each target is located inside `$$tag ...$$` symbols.
  There are 9 different types of
  tags (`code_by_description, conditional_statement, var_declaration, class_initialization, function_name, function_parameter, description_by_code, method_call, imports`)
  that can be later used for troubleshooting or analysis, but as of now it is just a small bonus since the dataset is
  relatively small.
  You can check example for each type
  by [data/tags_examples.txt](https://github.com/TyKo0707/code_completion/blob/main/data/tags_example.txt)
  Example:

```python 
# Example 1:
tensorboard_callback = keras.callbacks.$$method_call
TensorBoard(log_dir=logdir, histogram_freq=1)$$

# target part is `TensorBoard(log_dir=logdir, histogram_freq=1)` with tag method_call

# Example 2:
def generate_time_based_id():
    # Get the current time in the format YYYYMMDDHHMMSSFFF (year, month, day, hour, minute, second, millisecond)
    return $$code_by_description
    datetime.now().strftime("%Y%m%d%H%M%S%f")$$
    # target part is `code_by_description datetime.now().strftime("%Y%m%d%H%M%S%f")` with tag code_by_description
```

* Extracted all targets using regex along with suffixes and prefixes from the same file. The resulting dataset consists
  of 37 code snippets and can be found
  here: [data/python_dataset.csv](https://github.com/TyKo0707/code_completion/blob/main/data/python_dataset.csv).

### 2. Generation Code Completions

* Using dataset and 4 models, generate missing code parts and save them
  into [data/python_dataset_gen.csv](https://github.com/TyKo0707/code_completion/blob/main/data/python_dataset_gen.csv)

### 3. Evaluation

The most exciting part is here. I need to evaluate generated data manually and automatically and then compare different
metrics by task suitability.

#### Goals of the Evaluation

For each generated sample, with its target form and context (prefix and suffix), we will measure the following:

- **Exact Match**: Does the generated code exactly match the target? If yes, skip further measurements.
- **Functional Correctness**: Does the generated code run successfully?
- **Syntactic Similarity**: How close is the generated code to the target, in terms of syntax?
- **Semantic Similarity**: Even if the code isn't identical, does it accomplish the task correctly, given the context?

#### Manual Evaluation Metrics

The arrows (&uarr; & &darr;) indicate the change in which direction is an indicator of improvement for each metric. <br>
I’ve defined three manual metrics to assess code generation:

- **Functional Correctness** &uarr;: Does the generated code compile and run without errors?
- **Factual Correctness** &uarr;: Does the generated code solve the intended task?
- **Relevance** &uarr;: How well does the generated code fit the context and solve the task? (This is subjective but
  important.)

#### Automatic Evaluation Metrics

I applied similar logic to automatic metrics, with some slight differences in focus:

- **Exact Match** &uarr;: (Exact match) Checks if the generated code is identical to the reference.
- **CHRF3** &uarr;: (Syntactic similarity) Evaluates character-level similarity, capturing fine details.
- **Edit Distance** &darr;: (Syntactic similarity) Counts the number of changes required to match the reference.
- **Embedding Similarity** &uarr;: (Semantic similarity) Measures how semantically similar the generated code is to the
  target using embeddings like [CodeT5p-110m-Embedding](https://huggingface.co/Salesforce/codet5p-110m-embedding).
- **ROUGE-L** &uarr;: (Syntactic similarity) Focuses on matching long subsequences between the generated and target
  code.
- **Function Correctness (LLM)** &uarr;: (Functional correctness) Automatically checks whether the generated code runs
  without errors, using LLMs for efficiency instead of manual tests.

*Note*: I was thinking about writing tests for completion code automatically.
But it happened to be quite tricky to cover all types of generated code.
So I decided to use LLM (Claude 3 Opus, because it seems that GPT-4 does not work properly, but you can try to use both
of them)
for this task, and the model has shown itself to be of sufficient quality and suitability for this.
You can see structure of a prompt
at [utils/prompt_gen.py](https://github.com/TyKo0707/code_completion/blob/main/utils/prompt_gen.py).

I will discuss their pros and cons later, now we are done with choosing the metrics.
Example of evaluation:

```python 
target: BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

generated: BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Manual
functional_correctness: 1;
factual_correctness: 0.8;
relevance: 0.9

# Automatic
exact_match: 0;
chrf3: 0.675;
edit_distance: 62;
embedding_similarity: 0.959;
rouge_l: 0.680;
function_correctness_llm: 1
```

### 4. Results

As a result of evaluating the generation of each of the models,
I obtained the results and stored them in
the [data/eval_results](https://github.com/TyKo0707/code_completion/tree/main/data/eval_results) directory.

#### Model Results

> Manual metrics
![manual_metrics](https://github.com/user-attachments/assets/b609575c-722c-4cf9-9aea-dfb5ac7b52cc)

> Automatic metrics
![automatic_metrics](https://github.com/user-attachments/assets/e5d4c464-fb70-4cc8-bbbb-adb2c34aac24)

- **Starcoder2-7B** was the best performer based on metrics, while **tiny_starcoder_py** was the worst.
- This result is surprising: I expected **starcoder2-15B** to lead significantly, but its generation flaws affected
  this.
- **Starcoder2-15B** generates outputs with significant edit distance deviations and often produces unexpected code,
  which can be both a strength and a weakness because the quality of unexpected remains good.
- Overall, all models performed reasonably well for their sizes, but I would highlight starcoder2-7B as the model that
  most clearly fulfills its tasks.

#### Comparison of Metrics

> Spearman method correlation matrix
![metrics_correlation_spearman](https://github.com/user-attachments/assets/7f282bc2-a936-41c8-b5d4-37b6323cd770)

> Pearson method correlation matrix
![metrics_correlation_pearson](https://github.com/user-attachments/assets/adcae412-dfbc-4975-ba4d-dc12b35d2269)

*Note*: I transformed the edit_distance metric with MinMaxScaler and inverted the values (new_val = 1 - old_val).
Otherwise, it will always have a negative correlation with all of the metrics.

Insights from the Correlation Plot:

- The binary variables **functional_correctness** and **functional_correctness_llm** correlate more with each other than
  with continuous variables.
- **Functional_correctness** correlates highest with its LLM-generated counterpart, indicating that the LLM reliably
  assesses the generated code's functionality.
- **Factual_correctness** is closely related to **chrf3**, **embedding_similarity**, and **rouge_l**, which becomes
  evident when we realize that these automatic metrics assess syntactic and semantic similarity, which is what
  factual_correctness represents too.
- **Relevance** also shows high correlation with **factual_correctness**, though they differ in some justified cases,
  leading me to retain this metric.

Differences in Pearson and Spearman Correlation Matrices:

- Pearson correlation values are lower than Spearman’s, suggesting some relationships are nonlinear and better
  represented by curves.
  Thus, using both methods together can provide a more comprehensive understanding of variable relationships.

### 5. Analysis

Here I'll outline the pros and cons I can find, for all the metrics I've used.
I also want to discuss the best model by my judgment and by evaluation results and how these compare with each other.

#### Metrics Analysis

| Metric                             | Pros                                                                                                                                                                                                       | Cons                                                                                                                                                |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Exact Match**                    | + Great for understanding perfect outputs.                                                                                                                                                                 | - Too strict for varying output lengths and types.                                                                                                  |
| **ROUGE-L**                        | + Captures token order (longest common subsequence). <br> + Evaluates tokens and sequences for nuanced assessment.                                                                                         | - Limited contextual understanding. <br> - High recall may not imply good performance; irrelevant matches possible.                                 |
| **CHRF (n=3)**                     | + Sensitive to small syntax variations. <br> + Less sensitive to output length than word-based metrics.                                                                                                    | - Limited contextual understanding. <br> - Doesn’t fully capture token order importance. <br> - High recall but may lack precision in code quality. |
| **Edit Distance**                  | + Measures closeness, useful for typos and minor errors.                                                                                                                                                   | - Limited contextual understanding. <br> - Overly sensitive to small changes. <br> - Ignores token order importance.                                |
| **Embedding Similarity (CodeT5p)** | + Captures semantic similarity, identifying equivalent completions. <br> + Context-aware, factoring in relationships between code elements. <br> + Can integrate with clustering and classification tasks. | - Computationally expensive. <br> - Misses specific syntax details, focuses on high-level similarities.                                             |
| **Functional Correctness (LLMs)**  | + Great for understanding whether generated output will work alone.                                                                                                                                        | - Cannot compile code with generated part, therefore not always correct.                                                                            | 

I would say, that I'd use **ROUGE-L**, **CHRF**, **Embedding Similarity**, and **Functional Correctness** as the prior
metrics for code completion tasks.

#### Models Analysis

In my opinion: from the code generation I have seen and from evaluation results,
Starcoder2-7B has proven to be the best model.
It generates complete finished code with a good understanding of context
and doesn't try to augment it unnecessarily (as Starcoder2-15B does).
To put it simply, if I were to use this model to help me write code, I would be happy with it and have a good
understanding of how to use it.

## Conclusion

### Findings and Learnings

- Understood more about how code completion works and that there are even specially trained models for this (like
  StarCoder1/2 family).
- How to generate code by general context of a file (suffix and prefix) using different models.
- What types of tasks code completion can be useful for (described by tags).
- What metrics are more appropriate to use for code generation and which of them meet my personal requirements (
  comparison of manual and automatic metrics).
- Realized which model is the best among the StarCoder models and that I could even use it as an assistant in writing my
  own code (at least in Python)

### Future Work

### Final Words
