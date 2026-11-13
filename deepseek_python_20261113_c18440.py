# ===== 0. Загрузка данных =====
# !pip install palmerpenguins -q   # раскомментируйте при необходимости

import json, re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import mutual_info_classif

NB_PATH = '/kaggle/pinguins.ipynb'

def load_from_notebook(nb_path):
    """Пытается найти путь к CSV в ноутбуке и загрузить его. Возвращает DataFrame или None."""
    try:
        with open(nb_path, encoding='utf-8') as f:
            nb = json.load(f)
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        m = re.search(r"read_csv\(\s*['\"](.+?)['\"]", code)
        if not m:
            return None
        csv_in_nb = m.group(1)
        candidates = [
            '/kaggle/input/penguins/penguins.csv',
            os.path.normpath(os.path.join(os.path.dirname(nb_path), csv_in_nb)),
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path:
            print('Загружено из', csv_path)
            return pd.read_csv(csv_path)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None

df = load_from_notebook(NB_PATH)
if df is None:
    from palmerpenguins import load_penguins
    df = load_penguins()
    print('CSV не найден, загружен тот же датасет из пакета palmerpenguins')

num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
df = df.dropna(subset=num_cols).copy()
df['species'] = df['species'].astype(str)   # убираем category, чтобы не ловить FutureWarning в groupby

# ===== 1. Новые признаки =====
df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
df['mass_per_flipper'] = df['body_mass_g'] / df['flipper_length_mm']

print('\nСредние и стандартные отклонения новых признаков по видам:')
print(df.groupby('species')[['bill_ratio', 'mass_per_flipper']].agg(['mean', 'std']).round(2))

# ===== 2. Насколько каждый признак разделяет виды =====
features = num_cols + ['bill_ratio', 'mass_per_flipper']
y = df['species']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# MI считаем одним вызовом на все признаки — быстрее и без лишних пересчётов
mi_all = mutual_info_classif(df[features], y, discrete_features=False, random_state=42)
mi_by_feature = dict(zip(features, mi_all))

rows = []
for col in features:
    groups = [g[col].values for _, g in df.groupby('species', observed=True)]

    f_stat, _ = stats.f_oneway(*groups)
    grand_mean = df[col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((df[col] - grand_mean) ** 2).sum()
    eta2 = ss_between / ss_total

    h_stat, _ = stats.kruskal(*groups)

    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    acc = cross_val_score(tree, df[[col]], y, cv=cv,
                          scoring='balanced_accuracy').mean()

    rows.append({'признак': col, 'eta2': round(eta2, 3), 'F': round(f_stat, 1),
                 'Крускал H': round(h_stat, 1), 'MI': round(mi_by_feature[col], 3),
                 'точность (bal.)': round(acc, 3)})

result = pd.DataFrame(rows).sort_values('точность (bal.)', ascending=False)
print('\nРазделимость видов по одному признаку (чем больше, тем лучше):')
print(result.to_string(index=False))

# ===== 3. Какие пары видов различает каждый признак =====
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_sd == 0:
        return np.nan
    return abs(a.mean() - b.mean()) / pooled_sd

pairs = [('Adelie', 'Chinstrap'), ('Adelie', 'Gentoo'), ('Chinstrap', 'Gentoo')]
d_table = pd.DataFrame(
    {f'{a}–{b}': [cohen_d(df.loc[y == a, c], df.loc[y == b, c]) for c in features]
     for a, b in pairs},
    index=features).round(2)
print('\nЭффект Коэна d для каждой пары видов:')
print(d_table)

# ===== 4. Лучший признак: пороги и ошибки =====
best = result.iloc[0]['признак']
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
pred = cross_val_predict(tree, df[[best]], y, cv=cv)
print(f'\nМатрица ошибок для {best}:')
print(pd.crosstab(y, pred, rownames=['факт'], colnames=['прогноз']).reindex(
    index=sorted(y.unique()), columns=sorted(y.unique()), fill_value=0))
tree.fit(df[[best]], y)
print('\nПороги дерева:')
print(export_text(tree, feature_names=[best]))

# ===== 5. Графики =====
palette = {'Adelie': '#1f77b4', 'Chinstrap': '#ff7f0e', 'Gentoo': '#2ca02c'}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.boxplot(data=df, x='species', y='bill_ratio', hue='species', palette=palette, ax=axes[0], legend=False)
axes[0].set_title('Длина клюва / высота клюва')
axes[0].set_ylabel('Отношение')

sns.boxplot(data=df, x='species', y='mass_per_flipper', hue='species', palette=palette, ax=axes[1], legend=False)
axes[1].set_title('Масса / длина ласта')
axes[1].set_ylabel('г на мм ласта')

sns.scatterplot(data=df, x='bill_ratio', y='mass_per_flipper', hue='species', palette=palette, ax=axes[2])
axes[2].set_title('Оба новых признака вместе')
axes[2].set_xlabel('Длина клюва / высота клюва')
axes[2].set_ylabel('г на мм ласта')
axes[2].legend(title='Вид')

for ax in axes[:2]:
    ax.set_xlabel('Вид')
plt.tight_layout()
plt.show()