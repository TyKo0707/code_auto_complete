# Code completion

The project's focus is on code completion. The pipeline separates code samples into three sections: prefix (before the cursor), middle (missing code), and suffix (after the cursor). The middle section's completions are generated using pre-trained open-source models: tiny_starcoder_py, starcoder2_3b, starcoder2_7b, and starcoder2_15b.

After creating completions, the outputs are examined both manually and automatically using the target code.

The goal is to establish which metrics correspond best with human evaluation and to assess the quality of code completion across various models.
