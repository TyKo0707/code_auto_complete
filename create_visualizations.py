import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from environs import Env

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")
models = env.list("MODELS_LIST")

merged_main = pd.DataFrame()

for i in range(len(models)):
    model = models[i]
    df_manual = pd.read_csv(f'{DATA_PATH}/eval_results/manual/manual_results_{model}.csv')
    df_automatic = pd.read_csv(f'{DATA_PATH}/eval_results/automatic/automatic_results_{model}_claude.csv')
    df_gen_tags = pd.read_csv(f'./data/python_dataset_gen.csv').tag

    merged_df = pd.merge(df_manual, df_automatic, left_index=True, right_index=True)
    merged_df.drop(columns=['chrf3_precision', 'chrf3_recall'], inplace=True)
    merged_df['functional_correctness'] = merged_df['functional_correctness'].astype(int)

    statistics = merged_df.describe()
    statistics.to_csv(f'{DATA_PATH}/statistics/statistics_{model}.csv')

    if i == 0:
        merged_main = merged_df
    else:
        merged_main = merged_main._append(merged_df, ignore_index=True)

# Correlation Plot
scaler = MinMaxScaler()
# Scale and inverse edit_distance to avoid negative correlation
merged_main['edit_distance'] = scaler.fit_transform(merged_main[['edit_distance']])
merged_main['edit_distance'] = 1 - merged_main['edit_distance']

for corr_method in ['spearman', 'pearson']:
    correlation_matrix = merged_main.corr(method=corr_method)

    plt.figure(figsize=(16, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.xticks(rotation=60)
    plt.title(f'Correlation Matrix of Metrics ({corr_method.upper()})')
    plt.savefig(f'{DATA_PATH}/plots/metrics_correlation_{corr_method}.png', dpi=300, bbox_inches='tight')
    plt.show()


# Evaluation Results Plot
models = {name: pd.read_csv(f'{DATA_PATH}/statistics/statistics_{name}.csv', index_col=0) for name in models}
metrics = {
    'manual': ['functional_correctness', 'factual_correctness', 'relevance'],
    'automatic': ['exact_match', 'chrf3', 'rouge_l', 'embedding_similarity', 'function_correctness_llm', 'edit_distance']
}

sns.set(style="whitegrid")


def plot_metrics(metrics, plot_type):
    mean_df = pd.DataFrame({name: model.loc['mean', metrics] for name, model in models.items()})

    if plot_type == 'automatic':
        mean_df.loc['edit_distance'] /= mean_df.loc['edit_distance'].max()
        figsize, width = (16, 9), 0.7
    else:
        figsize, width = (10, 6), 0.5

    mean_df = mean_df.reset_index().melt(id_vars='index', var_name='model', value_name='score')

    plt.figure(figsize=figsize)
    sns.barplot(x='model', y='score', hue='index', data=mean_df, palette='muted', width=width)
    plt.title(f'{plot_type.capitalize()} Metrics for Each Model')
    plt.ylabel('Mean Scores')
    plt.xlabel('Models')
    plt.xticks(rotation=0)
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{DATA_PATH}/plots/{plot_type}_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


# Plot metrics
for plot_type in metrics:
    plot_metrics(metrics[plot_type], plot_type)