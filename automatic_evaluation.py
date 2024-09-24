import pandas as pd
from nltk.translate.chrf_score import chrf_precision_recall_fscore_support
from rouge_score import rouge_scorer
from nltk import edit_distance
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from utils import models_api, prompt_gen
from environs import Env

env = Env()
env.read_env()

EMBEDDING_CHECKPOINT = env.str("EMBEDDING_CHECKPOINT")

# Use model for embeddings generation for code
tokenizer_embeddings = AutoTokenizer.from_pretrained(EMBEDDING_CHECKPOINT, trust_remote_code=True)
model_embeddings = AutoModel.from_pretrained(EMBEDDING_CHECKPOINT, trust_remote_code=True)

# Use default starcoder tokenizer for computing ROUGE-L score
tokenizer_rouge = AutoTokenizer.from_pretrained('bigcode/tiny_starcoder_py')
rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=tokenizer_rouge)


# Function to compute exact match
def exact_match(golden, generated):
    return 1 if golden == generated else 0


# Function to compute chrF (precision, recall, fscore)
def compute_chrf(golden, generated):
    precision, recall, fscore, _ = chrf_precision_recall_fscore_support(golden, generated, 3)
    return precision, recall, fscore


# Function to compute Levenshtein (Edit) Distance
def compute_edit_distance(golden, generated):
    return edit_distance(golden, generated)


# Function to compute ROUGE score
def compute_rouge(golden, generated):
    return rouge_scorer.score(golden, generated)['rougeL'].fmeasure


# Function to compute embedding similarity (cosine similarity between embeddings)
def compute_embedding_similarity(golden, generated):
    golden_embedding = model_embeddings(tokenizer_embeddings.encode(golden, return_tensors="pt"))[0].detach().numpy()
    generated_embedding = model_embeddings(tokenizer_embeddings.encode(generated, return_tensors="pt"))[
        0].detach().numpy()
    similarity = round(cosine_similarity([golden_embedding], [generated_embedding])[0][0], 5)
    print(f'Golden: {golden}\nGen: {generated}\nSimilarity: {similarity}')
    return similarity


def evaluation_step(prefix, suffix, target, generated):
    prompt = prompt_gen.generate_prompt_function_correctness(prefix, suffix, generated)
    response = models_api.call_anthropic_api(prompt).strip()
    functional_correctness = None
    # Give 3 tries for model to generate output in valid format
    for i in range(3):
        try:
            functional_correctness = int(response[0])
            break
        except:
            print(f'The model generated an invalid output! {i + 1} try.')

    if functional_correctness is None:
        raise Exception(f"An unexpected error occurred while generating output!")

    chrf_precision, chrf_recall, chrf_fscore = compute_chrf(target, generated)

    result = {
        'exact_match': exact_match(target, generated),
        'chrf3_precision`': chrf_precision,
        'chrf3_recall': chrf_recall,
        'chrf3': chrf_fscore,
        'edit_distance': compute_edit_distance(target, generated),
        'embedding_similarity': compute_embedding_similarity(target, generated),
        'rouge_l': compute_rouge(target, generated),
        'function_correctness_llm': functional_correctness
    }
    return result


if __name__ == '__main__':
    DATA_PATH = env.str("DATA_PATH")
    models = env.list("MODELS_LIST")

    df = pd.read_csv(f'{DATA_PATH}/python_dataset_gen.csv')
    for model in models:
        save_path = f'{DATA_PATH}/eval_results/automatic/automatic_results_{model}.csv'
        results = []

        for ind, row in df.iterrows():
            generated = row[f'gen_{model}']
            result = evaluation_step(row.prefix, row.suffix, row.content, generated)
            results.append(result)

            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
