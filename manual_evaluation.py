from utils.utils import color


def evaluation_step(prefix, suffix, target, tag, generated, index, model_name):
    print(f'Index: {color(index + 1)}\n'
          f'Generated output ({model_name}): \n{color(generated)}\n'
          f'Code: {prefix[-250:].lstrip()}{color(target)}{suffix[:250].rstrip()}\n'
          f'Tag: {color(tag)}\n')

    print('Rate the output by four criteria')
    functional_correctness = bool(
        int(input('Functional Correctness - will generated line compile? (bool int: 0/1): ') or "1")
    )
    factual_correctness = float(input('Factual Correctness - is generated code correct? (int: 0-10): ') or "10") / 10
    relevance = float(input('Relevance (int: 0-10): ') or "10") / 10
    results = {'functional_correctness': functional_correctness,
               'factual_correctness': factual_correctness,
               'relevance': relevance}
    return results


if __name__ == '__main__':
    import pandas as pd
    from environs import Env

    env = Env()
    env.read_env()
    DATA_PATH = env.str("DATA_PATH")
    models = env.list("MODELS_LIST")

    df = pd.read_csv(f'{DATA_PATH}/python_dataset_gen.csv')

    for model in models:
        save_path = f'{DATA_PATH}/eval_results/manual/manual_results_{model}.csv'
        results = []

        for ind, row in df.iterrows():
            generated = row[f'gen_{model}']
            result = evaluation_step(row.prefix, row.suffix, row.content, row.tag, generated, ind, model)
            results.append(result)

            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
