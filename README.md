# Code completion

The project's focus is on code completion. The pipeline separates code samples into three sections: prefix (before the cursor), middle (missing code), and suffix (after the cursor). 
The middle section's completions are generated using pre-trained open-source models: `tiny_starcoder_py`, `starcoder2_3b`, `starcoder2_7b`, and `starcoder2_15b`.

After creating completions, the outputs are examined both manually and automatically using the target code.

The goal is to establish which metrics correspond best with human evaluation and to assess the quality of code completion across various models.

## Steps of solution
0. **Exploring the models**:
* Explore different models ([explore_tiny_starcoder.ipynb](https://github.com/TyKo0707/code_completion/blob/main/models_explore/explore_tiny_starcoder.ipynb), [explore_big_starcoder.ipynb](https://github.com/TyKo0707/code_completion/blob/main/models_explore/explore_big_starcoder.ipynb)) specified for code generation to find the most suitable for the task.
* Find the optimal config and method of generation for each model.
* Result of this step: decided to use starcoder and starcoder-based models `tiny_starcoder_py`, `starcoder2_3b`, `starcoder2_7b`, and `starcoder2_15b`.


1. **Data Collection and Preprocessing**:
I need to create a dataset consists of prefix, middle part (target), and suffix.
* Took a few files from the local repository and annotated them.
Each target is located inside `$$tag ...$$` symbols, so it looks like this:
```python 
tensorboard_callback = keras.callbacks.$$method_call TensorBoard(log_dir=logdir, histogram_freq=1)$$
# target part is `TensorBoard(log_dir=logdir, histogram_freq=1)`.
``` 
