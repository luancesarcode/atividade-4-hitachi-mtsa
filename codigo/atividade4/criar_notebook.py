"""Gera o notebook final da Atividade 4, autocontido para o Colab.

Executar da raiz: python -m atividade4.criar_notebook
Embute os arquivos necessários do tutorial e da atividade, mais o CSV
compactado, com SHA-256. Gera o notebook sem saídas; quem executa é
atividade4/validar_notebook.py.
"""
import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'a5959467eca45f930341783a7006affcae5e6e88'
REPO = 'https://github.com/CristoferSilva/mtsa-tutorial-compet'
CSV = 'examples/linear_ts_with_confounder_n5000_vars6_lag2_gaussian.csv'
NOME = 'Luan_Cesar_de_Souza_Nascimento_Atividade_4.ipynb'
TARGETS = [ROOT.parent / 'saida' / NOME, ROOT / 'examples' / NOME]

# Arquivos do tutorial usados sem alteração (conferidos contra o commit).
FRAMEWORK = ['mtsa/__init__.py', 'mtsa/loss.py', 'mtsa/trainer.py', 'mtsa/predictor.py',
             'mtsa/losses/mse_loss.py', 'mtsa/models/__init__.py', 'mtsa/models/hitachi.py',
             'mtsa/trainers/unsupervised_trainer.py', 'mtsa/predictors/standard_predictor.py',
             'mtsa/utils/__init__.py', 'mtsa/utils/data.py']
# Arquivos criados ou alterados nesta atividade, mais a fonte original preservada.
ATIVIDADE = ['examples/sample_data_utils.py', 'mtsa/predictors/hitachi_score.py',
             'atividade4/__init__.py', 'atividade4/experimento.py',
             'atividade4/fontes/original_sample_data_utils.py', 'atividade4/fontes/proveniencia.json',
             'tests/test_atividade4.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def same_as_commit(relative):
    original = subprocess.run(['git', 'show', f'{COMMIT}:{relative}'], cwd=ROOT,
                              capture_output=True, check=True).stdout
    # Compara texto com fim de linha normalizado (o checkout no Windows usa CRLF).
    local = (ROOT / relative).read_bytes()
    return original.replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n')


def bundle():
    files, origin = {}, {}
    for relative in FRAMEWORK + ATIVIDADE:
        files[relative] = (ROOT / relative).read_text(encoding='utf-8')
        if relative in FRAMEWORK:
            if not same_as_commit(relative):
                raise RuntimeError(f'{relative} difere do commit {COMMIT[:7]}.')
            origin[relative] = f'tutorial, idêntico ao commit {COMMIT[:7]}'
        elif relative == 'atividade4/fontes/original_sample_data_utils.py':
            origin[relative] = f'cópia de examples/sample_data_utils.py no commit {COMMIT[:7]}'
        else:
            origin[relative] = 'criado ou alterado na Atividade 4'
    hashes = {k: sha(v.encode('utf-8')) for k, v in files.items()}
    raw = (ROOT / CSV).read_bytes()
    csv_b64 = base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')
    return files, hashes, origin, csv_b64, sha(raw)


def expected_results():
    report = json.loads((ROOT / 'atividade4/resultados/resultados.json').read_text(encoding='utf-8'))
    return {'f1_mean': report['summary']['f1']['mean'], 'f1_sd': report['summary']['f1']['sd'],
            'baseline_f1': report['baseline']['f1'],
            'f1_by_seed': {str(r['seed']): r['f1'] for r in report['runs']},
            'environment': report['environment']}


def build():
    files, hashes, origin, csv_b64, csv_sha = bundle()
    expected = expected_results()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    hidden = {'cellView': 'form', 'jupyter': {'source_hidden': True}}
    cells = []
    add = cells.append

    add(md(f'''# Atividade 4: correção dos dados e Hitachi no MTSA

**Luan César de Souza Nascimento** · estudante de Engenharia de Controle e Automação (UFPE) · COMPET Moura · 21/10/2026

| Campo | Informação |
|---|---|
| Repositório usado | [{REPO.removeprefix('https://')}]({REPO}), commit `{COMMIT}` |
| Parte I | encontrar o problema em `get_tabular_sample_data()` |
| Parte II | treinar e avaliar o Hitachi no framework do tutorial, como no exemplo do GANF |
| Dados | CSV sintético do tutorial (5.000 linhas, 6 variáveis); não são dados da Moura |

Para rodar no Colab: **Ambiente de execução → Executar tudo**. Os módulos e o CSV estão embutidos no notebook, então não é preciso enviar o repositório nem baixar nada. A execução completa leva cerca de um minuto em CPU.

### Índice
1. [Ambiente e configuração](#ambiente)
2. [Origem dos dados e inspeção](#dados)
3. [Parte I: o erro em `get_tabular_sample_data()`](#parte1)
4. [Protocolo de avaliação e baseline](#protocolo)
5. [Parte II: Hitachi no framework](#parte2)
6. [Resultados por seed](#resultados)
7. [Gráficos e interpretação](#graficos)
8. [Limitações](#limites)
9. [Testes e reprodução](#testes)
10. [Respostas para a apresentação e referências](#apresentacao)'''))

    # 1. Ambiente
    add(md('''## 1. Ambiente e configuração <a id="ambiente" name="ambiente"></a>

Entrada: nada além deste notebook. Saída: uma pasta de trabalho com os módulos do tutorial, os da atividade e o CSV, todos conferidos por SHA-256.

Usamos só pacotes que já vêm no Colab (PyTorch, NumPy, pandas, scikit-learn, Matplotlib, tqdm). O pacote `mtsa` do PyPI tem outra interface; por isso o notebook usa a cópia do tutorial, e não `pip install mtsa`.'''))
    add(code('''import hashlib
import importlib.util
import os
import sys
from pathlib import Path

os.environ['TQDM_DISABLE'] = '1'  # barras de progresso do treinador poluem a saída salva

import numpy as np
import pandas as pd
import sklearn
import torch
from IPython.display import Code, Markdown, display

EM_COLAB = 'google.colab' in sys.modules
PASTA = Path('/content/atividade4_mtsa') if EM_COLAB else Path.cwd() / 'atividade4_mtsa'
PASTA.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'item': ['Colab', 'Python', 'PyTorch', 'NumPy', 'pandas', 'scikit-learn', 'pasta de trabalho'],
              'valor': [EM_COLAB, sys.version.split()[0], torch.__version__, np.__version__,
                        pd.__version__, sklearn.__version__, str(PASTA)]})'''))
    src = ('# @title Arquivos embutidos (recolhido; abra para inspecionar)\n'
           '# Cópia literal dos arquivos; os hashes foram calculados sobre os bytes originais.\n'
           f'SOURCE_FILES = {json.dumps(files, ensure_ascii=False, indent=1)}\n'
           f'SOURCE_HASHES = {json.dumps(hashes, indent=1)}\n'
           f'SOURCE_ORIGIN = {json.dumps(origin, ensure_ascii=False, indent=1)}\n'
           f'CSV_PATH = {CSV!r}\n'
           f'CSV_SHA256 = {csv_sha!r}\n'
           f'CSV_GZIP_B64 = {csv_b64!r}\n')
    add(code(src, metadata=hidden))
    add(code('''import base64
import gzip


def gravar_arquivos(destino):
    """Grava os arquivos embutidos e confere o SHA-256 de cada um."""
    linhas = []
    for relativo, texto in SOURCE_FILES.items():
        caminho = destino / relativo
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(texto.encode('utf-8'))
        obtido = hashlib.sha256(caminho.read_bytes()).hexdigest()
        assert obtido == SOURCE_HASHES[relativo], f'Hash divergente: {relativo}'
        linhas.append((relativo, SOURCE_ORIGIN[relativo], obtido[:12]))
    dados = gzip.decompress(base64.b64decode(CSV_GZIP_B64))
    assert hashlib.sha256(dados).hexdigest() == CSV_SHA256, 'Hash divergente no CSV'
    (destino / CSV_PATH).write_bytes(dados)
    linhas.append((CSV_PATH, 'tutorial, dados sintéticos', CSV_SHA256[:12]))
    return pd.DataFrame(linhas, columns=['arquivo', 'origem', 'sha256 (início)'])


arquivos = gravar_arquivos(PASTA)
if str(PASTA) not in sys.path:
    sys.path.insert(0, str(PASTA))
os.chdir(PASTA)
arquivos'''))
    add(md('''A configuração do experimento fica em um único lugar, `CONFIG` em `atividade4/experimento.py`. O notebook e a linha de comando usam as mesmas funções. Os valores foram fixados antes de olhar o teste e não houve busca de hiperparâmetros. Se alguém mudar algo aqui, os números deixam de corresponder aos registrados no parecer.'''))
    add(code('''import atividade4.experimento as exp

torch.set_num_threads(exp.CONFIG['torch_threads'])
torch.use_deterministic_algorithms(True)
CONFIG = exp.CONFIG
pd.Series(CONFIG, name='valor').to_frame()'''))

    # 2. Dados
    add(md('''## 2. Origem dos dados e inspeção <a id="dados" name="dados"></a>

O CSV `linear_ts_with_confounder_n5000_vars6_lag2_gaussian.csv` vem do próprio tutorial. Pelo nome, é uma série linear sintética com ruído gaussiano, seis variáveis `X1`…`X6` e um confundidor `U`. O carregador descarta `U` e `time`, então o modelo vê seis entradas. As anomalias não existem no arquivo: são somadas pelo código, como mostra a Parte I.'''))
    add(code('''df = pd.read_csv(CSV_PATH)
print('linhas, colunas:', df.shape)
display(df.head(3))
X_bruto = df.drop(columns=['U', 'time']).to_numpy()
resumo = pd.DataFrame({'média': X_bruto.mean(0), 'desvio': X_bruto.std(0),
                       'autocorrelação lag 1': [np.corrcoef(X_bruto[:-1, j], X_bruto[1:, j])[0, 1] for j in range(6)]},
                      index=[f'X{j + 1}' for j in range(6)])
resumo.round(3)'''))
    add(md('''As variáveis têm média perto de zero, desvio em torno de 0,1 e quase nenhuma autocorrelação de um passo. Na prática, cada linha se comporta como uma amostra independente. Isso explica por que um modelo de janela 1 funciona bem aqui e por que a ordem temporal pesa mais para modelos com janelas, como o GANF.'''))

    # 3. Parte I
    add(md('''## 3. Parte I: o erro em `get_tabular_sample_data()` <a id="parte1" name="parte1"></a>

A função interna `set_abnormal_data` soma um deslocamento a quatro intervalos de linhas. O trecho original, sem alteração:'''))
    add(code('''ORIGINAL = Path('atividade4/fontes/original_sample_data_utils.py').read_text(encoding='utf-8')
inicio = ORIGINAL.index('    def set_abnormal_data')
display(Code(ORIGINAL[inicio:ORIGINAL.index('        return x, y_test', inicio)], language='python'))'''))
    add(md('''Os três últimos intervalos usam `step` como duração: `(2*step, 3*step)`, `(4*step, 5*step)` e `(n-100-step, n-100)`. O primeiro usa `step` como fim absoluto: `(start_point or step, step)`. A função é chamada duas vezes:

- `step=200, start_point=150` → `[150:200]`, só 50 linhas em vez de 200;
- `step=13, start_point=200` → `[200:13]`, fatia vazia. Nada é alterado e nenhum erro aparece.

O `or` também trata `start_point=0` como ausente, e sem início explícito o intervalo seria `[step:step]`, vazio. A tabela abaixo recalcula os intervalos com o comprimento real do arquivo.'''))
    add(code('''def intervalos(step, inicio, n, corrigido):
    primeiro = (step if inicio is None else inicio) if corrigido else (inicio or step)
    fim = primeiro + step if corrigido else step
    return [(primeiro, fim), (2 * step, 3 * step), (4 * step, 5 * step), (n - (100 + step), n - 100)]


linhas = []
for step, inicio in [(200, 150), (13, 200)]:
    for k, (orig, novo) in enumerate(zip(intervalos(step, inicio, 5000, False), intervalos(step, inicio, 5000, True)), 1):
        linhas.append({'chamada': f'step={step}, início={inicio}', 'intervalo': k,
                       'original': f'[{orig[0]}:{orig[1]}]', 'linhas originais': max(0, orig[1] - orig[0]),
                       'corrigido': f'[{novo[0]}:{novo[1]}]', 'linhas corrigidas': novo[1] - novo[0]})
pd.DataFrame(linhas)'''))
    add(md('''Para medir o efeito nos dados, e não só no texto, executamos a função original sobre o CSV. Só duas linhas foram adaptadas para rodar fora da pasta `examples/`: o caminho do CSV e uma importação de áudio que essa função não usa. A versão corrigida roda no modo `legacy`, que mantém a montagem original do teste para isolar o efeito do intervalo.'''))
    add(code('''import contextlib
import io

adaptado = (ORIGINAL.replace('from mtsa import files_train_test_split', '')
                    .replace('os.path.join(os.getcwd()) + "/linear', '"examples/linear'))
espaco = {}
exec(compile(adaptado, 'original_sample_data_utils.py', 'exec'), espaco)
with contextlib.redirect_stdout(io.StringIO()):  # a original imprime contagens
    Xtr_o, ytr_o, Xte_o, yte_o = espaco['get_tabular_sample_data']()

from examples.sample_data_utils import get_tabular_sample_data, set_abnormal_data

Xtr_l, ytr_l, Xte_l, yte_l, meta_l = get_tabular_sample_data(split_mode='legacy', return_metadata=True)
pd.DataFrame({'original': [int(yte_o.sum()), len(Xte_o), len(Xtr_o)],
              'intervalo corrigido (legacy)': [int(yte_l.sum()), len(Xte_l), len(Xtr_l)]},
             index=['linhas anômalas', 'linhas de teste', 'linhas de treino'])'''))
    add(md('''Com o erro, o teste tem 676 linhas anômalas em vez de 826. As 150 que faltam são `[200:350]`, o resto da primeira faixa da chamada com `step=200`. A faixa vazia `[200:213]` da outra chamada cai dentro desse trecho, por isso não soma linhas novas, mas deixa essas 13 linhas sem o deslocamento em X1 e X2. O código roda normalmente, e o erro passa despercebido.

A correção, em `examples/sample_data_utils.py`, usa início explícito, fim `início + duração` e aceita início zero. Intervalos vazios ou fora dos dados agora geram `ValueError`.'''))
    add(code('''import inspect

display(Code(inspect.getsource(set_abnormal_data), language='python'))
_, y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), features=[0], start_point=200, magnitude=7)
assert y[200:213].all() and y[213] == 0
_, y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), start_point=0)
assert y[:13].all() and y[13] == 0
try:
    set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000), start_point=995)
except ValueError as erro:
    print('Intervalo fora dos dados agora falha:', erro)'''))
    add(md('''Outros pontos encontrados na mesma função:

- `magnitude * sin(pi/4)` é uma constante. A anomalia é um degrau de 1,41 (magnitude 2) ou 4,95 (magnitude 7), não uma senoide no tempo.
- A original chamava `np.random.seed(42)`, alterando o gerador global de quem a importasse, sem usar números aleatórios. Foi removido.
- O caminho do CSV dependia da pasta atual (`os.getcwd()`); agora é relativo ao arquivo.
- As duas chamadas se sobrepõem em `[200:213]` e no fim da série. Nessas linhas, quatro variáveis ficam deslocadas. O rótulo continua 1, o que está correto, mas vale saber que existem.'''))
    add(md('''### Montagem do teste: perda da ordem temporal

Depois da injeção, a original separa as linhas por classe e monta o teste como `vstack([primeiros 826 normais, todos os anômalos])`. O treino recebe os normais restantes. Isso não coloca a mesma linha em treino e teste, então não é vazamento de rótulo. O problema é outro: a sequência deixa de ser contínua.

Esta entrega usa outro protocolo, `chronological`: as primeiras 3.000 linhas, sem anomalias, servem para treino e validação, e as últimas 2.000 formam o teste, com anomalias inseridas só nele. É uma mudança metodológica, separada da correção do intervalo. A célula abaixo conta as emendas, isto é, pontos em que linhas vizinhas no teste não são vizinhas no tempo, e quantas janelas de 12 linhas (tamanho usado no exemplo do GANF) atravessam alguma emenda.'''))
    add(code('''def emendas(indices, janela=12):
    indices = np.asarray(indices)
    quebras = np.flatnonzero(np.diff(indices) != 1)
    inicios = np.arange(len(indices) - janela + 1)
    cruzam = sum(bool(((quebras >= s) & (quebras < s + janela - 1)).any()) for s in inicios)
    return len(quebras), cruzam, len(inicios)


Xdev, ydev, Xte, yte, meta = get_tabular_sample_data(return_metadata=True)
tabela = []
for nome, m in [('legacy (original, intervalo corrigido)', meta_l), ('chronological (entrega)', meta)]:
    q, c, total = emendas(m['test_indices'])
    treino_depois = np.mean(np.asarray(m['train_indices']) > min(m['test_indices']))
    tabela.append({'montagem': nome, 'emendas no teste': q, 'janelas de 12 que cruzam emendas': f'{c} de {total}',
                   'treino posterior ao início do teste': f'{treino_depois:.0%}'})
pd.DataFrame(tabela)'''))
    add(md('''No `legacy`, um modelo de janelas vê trechos colados que nunca foram vizinhos, inclusive a passagem abrupta de normal para anômalo, e treina com linhas posteriores ao teste. No `chronological`, o teste é um bloco contínuo e o treino vem inteiro antes dele. Como esta série quase não tem autocorrelação, o efeito para um modelo de janela 1 é pequeno; para o GANF e outros modelos temporais, pode mudar o escore.'''))

    # 4. Protocolo
    add(md('''## 4. Protocolo de avaliação e baseline <a id="protocolo" name="protocolo"></a>

Entrada: as 5.000 linhas. Saída: três conjuntos padronizados.

| Conjunto | Linhas do CSV | Uso |
|---|---|---|
| Treino | 0-2499, normais | ajuste do modelo e do `StandardScaler` |
| Validação | 2500-2999, normais | só o limiar: percentil 99 dos escores |
| Teste | 3000-4999, 1.174 normais e 826 anômalas | avaliação final, uma vez por seed |

O limiar vem só de dados normais da validação. Por construção, cerca de 1% dos normais fica acima dele, então espera-se uma taxa de falso alarme perto de 1%.

Métricas usadas: ROC-AUC e average precision medem a ordenação dos escores, sem limiar; F1, precisão, recall e FAR (taxa de falso alarme, FP/(FP+TN)) dependem do limiar. A classe positiva é a anomalia (rótulo 1), e escore maior significa mais anômalo.

O baseline é uma regra sem treino: a média dos quadrados das seis variáveis padronizadas, com o mesmo limiar pela validação. Se ele já acerta quase tudo, o conjunto é fácil e nota alta do Hitachi diz pouco.'''))
    add(code('''data = exp.prepare_data(CONFIG)
assert data['yfit'].sum() == 0 and data['yval'].sum() == 0
pd.DataFrame({'linhas': [len(data['fit']), len(data['val']), len(data['test'])],
              'anomalias': [int(data['yfit'].sum()), int(data['yval'].sum()), int(data['ytest'].sum())],
              'média após padronizar': [data[k].mean().round(3) for k in ['fit', 'val', 'test']]},
             index=['treino', 'validação', 'teste'])'''))
    add(code('''base, escore_base = exp.baseline(data, CONFIG)
pd.Series(base).to_frame('baseline').T[['roc_auc', 'average_precision', 'f1', 'recall', 'far', 'tp', 'fp', 'fn', 'tn']]'''))
    add(md('''Para ter uma ideia da dificuldade, a tabela abaixo mede o tamanho dos deslocamentos em desvios-padrão de cada variável no treino.'''))
    add(code('''escala = data['scaler'].scale_
pd.DataFrame({'variáveis': ['X5, X6 (magnitude 2)', 'X1, X2 (magnitude 7)'],
              'deslocamento': [2 * np.sin(np.pi / 4), 7 * np.sin(np.pi / 4)],
              'em desvios-padrão': [(2 * np.sin(np.pi / 4) / escala[[4, 5]]).round(1).tolist(),
                                    (7 * np.sin(np.pi / 4) / escala[[0, 1]]).round(1).tolist()]})'''))

    # 5. Parte II
    add(md('''## 5. Parte II: Hitachi no framework <a id="parte2" name="parte2"></a>

A classe `Hitachi` já existia no tutorial (`mtsa/models/hitachi.py`), e foi usada sem alteração. A contribuição desta atividade é ligá-la ao mesmo fluxo do exemplo do GANF em `examples/MTSA.ipynb` e medir o desempenho. A arquitetura é 6 → 64 → 64 → 8 → 64 → 64 → 6, com ReLU nas camadas internas e saída linear. Com seis entradas, a camada de 8 unidades é maior que a entrada, então não há compressão dimensional.'''))
    add(code('''from mtsa.models.hitachi import Hitachi
from mtsa.predictors.hitachi_score import HitachiScore

display(Markdown('`mtsa/models/hitachi.py` (tutorial, sem alteração)'))
display(Code(inspect.getsource(Hitachi), language='python'))
display(Markdown('`mtsa/predictors/hitachi_score.py` (adaptador criado nesta atividade)'))
display(Code(inspect.getsource(HitachiScore), language='python'))'''))
    add(md('''O exemplo do GANF organiza o aprendizado em cinco partes. A correspondência para o Hitachi:

| Parte do framework | GANF no tutorial | Hitachi nesta atividade |
|---|---|---|
| Conjunto de hipóteses | `GANF(...)` | `Hitachi(input_dim=6)` |
| Medida de erro | `LagrangeLoss(ForwardLoss, TraceExponentialMatrixLoss)` | `MSELoss` com entrada = alvo |
| Otimizador | Adam | Adam, taxa 0,001 |
| Algoritmo de aprendizado | `LagrangeTrainer` | `UnsupervisedTrainer`, 30 épocas, só normais |
| Predição | `StandardPredictor` sobre o modelo | `StandardPredictor` sobre `HitachiScore` |

O GANF já devolve um escore por janela. O Hitachi devolve a reconstrução, então faltava uma peça que transformasse reconstrução em escore. É o `HitachiScore`: erro quadrático médio por janela, maior = mais anômalo. Ele também remove o canal extra que o `StandardPredictor` acrescenta. A célula abaixo mostra as dimensões em cada etapa.'''))
    add(code('''from mtsa.losses.mse_loss import MSELoss
from mtsa.predictors.standard_predictor import StandardPredictor
from mtsa.trainers.unsupervised_trainer import UnsupervisedTrainer
from mtsa.utils.data import build_window_tensor

modelo = Hitachi(input_dim=6)
lote, _ = next(iter(exp.loader(data['fit'], data['yfit'], config=CONFIG)))
janela_predicao = build_window_tensor(data['test'], 0, 1).unsqueeze(0)
pd.DataFrame({'tensor': ['lote do get_dataloader (treino)', 'saída do Hitachi', 'perda MSELoss',
                         'janela do StandardPredictor', 'escore do HitachiScore'],
              'forma': [tuple(lote.shape), tuple(modelo(lote).shape), tuple(MSELoss()(lote, lote, modelo)[0].shape),
                        tuple(janela_predicao.shape), tuple(HitachiScore(modelo).eval()(janela_predicao).shape)]})'''))
    add(md('''O treino de uma seed fica em `exp.train_one_seed`, reproduzido abaixo. Ele cria o modelo com a seed, treina com o `UnsupervisedTrainer`, calcula o limiar na validação e só então pontua o teste.'''))
    add(code('''display(Code(inspect.getsource(exp.train_one_seed), language='python'))'''))

    # 6. Resultados
    add(md('''## 6. Resultados por seed <a id="resultados" name="resultados"></a>

Cinco seeds fixadas antes da avaliação: 11, 22, 33, 42 e 55. A seed 42 é a que tem modelo e predições salvos, por ser a padrão do tutorial, e não a de melhor resultado. O desvio entre seeds mede só a variação do treinamento nesta divisão, não a incerteza em dados novos.'''))
    add(code('''registros, escores = [], {}
for seed in CONFIG['seeds']:
    registro, modelo_seed, _, escore_teste = exp.train_one_seed(seed, data, CONFIG)
    registros.append(registro)
    escores[seed] = escore_teste
por_seed = pd.DataFrame(registros).set_index('seed')
por_seed[['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'far', 'tp', 'fp', 'fn', 'tn', 'threshold']].round(6)'''))
    add(code('''resumo = pd.DataFrame(exp.summarize(registros)).T
resumo.columns = ['Hitachi: média', 'desvio entre seeds']
resumo['baseline'] = [base[k] for k in resumo.index]
resumo.round(6)'''))
    add(md('''O recall foi 1,0 em todas as seeds e também no baseline: nenhuma anomalia ficou abaixo do limiar. A diferença de F1 vem apenas dos falsos alarmes. O Hitachi teve de 7 a 18 falsos positivos entre 1.174 normais, e o baseline teve 15. Com essa variação entre seeds, a diferença não sustenta que o Hitachi seja melhor que a regra simples.'''))
    add(code('''saidas = PASTA / 'resultados_notebook'
saidas.mkdir(exist_ok=True)
por_seed.to_csv(saidas / 'metricas_seeds.csv')
pd.DataFrame({'linha_csv': meta['test_indices'], 'y_true': data['ytest'],
              'score_hitachi_seed42': escores[CONFIG['saved_seed']], 'score_baseline': escore_base}
             ).to_csv(saidas / 'predicoes_teste.csv', index=False)
print('Salvo em', saidas)'''))

    # 7. Gráficos
    add(md('''## 7. Gráficos e interpretação <a id="graficos" name="graficos"></a>

À esquerda, o escore do Hitachi (seed 42) ao longo do teste, em escala log, com o limiar e as faixas anômalas em cinza. À direita, a distribuição dos escores por classe. Os rótulos aparecem apenas para leitura do resultado; nada aqui é usado para escolher parâmetros.'''))
    add(code('''import matplotlib.pyplot as plt

s42 = escores[CONFIG['saved_seed']]
lim42 = por_seed.loc[CONFIG['saved_seed'], 'threshold']
linha = np.asarray(meta['test_indices'])
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [2, 1]})
a.fill_between(linha, 1e-5, 10, where=data['ytest'] == 1, color='0.85', step='mid', label='anomalia injetada')
a.plot(linha, s42, lw=0.7, color='#233175', label='escore Hitachi')
a.axhline(lim42, color='#F4AB1C', lw=2, label='limiar (p99 da validação)')
a.set(yscale='log', ylim=(1e-4, 10), xlabel='linha do CSV', ylabel='MSE de reconstrução', title='Teste, seed 42')
a.legend(loc='upper left', fontsize=8)
bins = np.logspace(-4, 1, 60)
b.hist(s42[data['ytest'] == 0], bins=bins, color='#233175', alpha=0.8, label='normal')
b.hist(s42[data['ytest'] == 1], bins=bins, color='#F4AB1C', alpha=0.9, label='anômalo')
b.axvline(lim42, color='black', ls='--', lw=1)
b.set(xscale='log', xlabel='escore', ylabel='linhas', title='Distribuição por classe')
b.legend(fontsize=8)
fig.tight_layout()
plt.show()'''))
    add(code('''margem = pd.DataFrame({
    'menor escore anômalo / limiar': [min(escores[s][data['ytest'] == 1]) / por_seed.loc[s, 'threshold'] for s in CONFIG['seeds']]
    + [escore_base[data['ytest'] == 1].min() / base['threshold']],
    'falsos positivos': list(por_seed['fp']) + [base['fp']]},
    index=[f'Hitachi seed {s}' for s in CONFIG['seeds']] + ['baseline'])
margem.round(1)'''))
    add(md('''Por que as notas são tão altas? Os deslocamentos ficam entre 10 e 46 desvios-padrão da variação normal. O menor escore anômalo fica de 2 a 10 vezes acima do limiar, no Hitachi e no baseline. Qualquer método razoável separa as classes.

Um detalhe chama atenção no gráfico: o erro do Hitachi nas anomalias fica perto de 0,1, embora o deslocamento seja de dezenas de desvios-padrão. A rede reconstrói boa parte do degrau, e sua margem sobre o limiar é menor que a do baseline. Isso combina com a ausência de gargalo, mas não testamos outra arquitetura para confirmar. Os erros que sobram são normais do teste que passaram do percentil 99 da validação, o que o próprio limiar já prevê.

O que o experimento mostra: o Hitachi treina e pontua dentro do framework do tutorial, com o sentido do escore correto e resultado estável entre seeds. O que ele não mostra: que o Hitachi seja melhor que o baseline ou que o GANF, ou que funcione em equipamento real.'''))

    # 8. Limitações
    add(md('''## 8. Limitações <a id="limites" name="limites"></a>

- Um único arquivo sintético e uma divisão temporal fixa. Não há dados reais nem outros conjuntos.
- As anomalias são degraus constantes e enormes. Desvios pequenos, deriva lenta ou ruído mudariam o quadro, e isso não foi testado.
- O GANF não foi reexecutado neste protocolo. Não há comparação Hitachi × GANF.
- O conjunto acústico não foi usado.
- A camada de 8 unidades não comprime seis entradas. O modelo detecta as anomalias porque elas ficam muito fora da faixa vista no treino. A seção 7 mostra que o erro nas anomalias é pequeno em termos absolutos; o mecanismo não foi investigado a fundo.
- No `StandardPredictor` do tutorial, o caminho `pointwise=False` falha (`y_hat_list = [], y_true_list = []` levanta `ValueError`). Usamos `pointwise=True`, que funciona; o arquivo do tutorial não foi alterado.
- Os números registrados vêm de Python 3.12, PyTorch 2.5.1 em CPU. Outra versão do PyTorch, como a do Colab, pode mudar alguns falsos positivos.'''))

    # 9. Testes
    add(md('''## 9. Testes e reprodução <a id="testes" name="testes"></a>

Os testes de `tests/test_atividade4.py` cobrem os intervalos, a separação dos conjuntos, o gerador aleatório, as métricas e a integração com treinador e preditor reais.'''))
    add(code('''import unittest

suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_atividade4')
resultado = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
assert resultado.wasSuccessful()'''))
    add(code(f'''# Resultados registrados na execução de referência (atividade4/resultados/resultados.json).
REFERENCIA = {json.dumps(expected, indent=1)}
f1_agora = {{str(s): float(por_seed.loc[s, 'f1']) for s in CONFIG['seeds']}}
iguais = all(abs(f1_agora[s] - REFERENCIA['f1_by_seed'][s]) < 1e-9 for s in f1_agora)
print('Ambiente de referência:', REFERENCIA['environment'])
print('Ambiente atual: python', sys.version.split()[0], '| torch', torch.__version__)
print('F1 por seed igual ao registrado:', iguais)
if not iguais:
    print('Diferença esperada se a versão do PyTorch ou o hardware mudarem. Compare a tabela da seção 6.')
assert abs(base['f1'] - REFERENCIA['baseline_f1']) < 1e-9, 'O baseline não depende do PyTorch e deveria ser idêntico.'
'''))
    add(md('''Para repetir fora do notebook, na raiz do repositório com as dependências de `requirements-atividade4.txt`:

```bash
python -m pytest tests/test_atividade4.py -q
python -m atividade4.experimento
```'''))

    # 10. Respostas
    add(md(f'''## 10. Respostas para a apresentação e referências <a id="apresentacao" name="apresentacao"></a>

**Qual foi a parte mais desafiadora?** Confiar nos dados antes de confiar no modelo. O erro do intervalo não quebra nada: o código roda e as métricas saem. Só aparece quando se conta quantas linhas foram de fato alteradas. Depois, foi preciso separar o que era correção do erro do que era mudança de protocolo (a montagem do teste) e entender por que qualquer método tirava nota perto de 1.

**Conclusão da Parte I.** O primeiro intervalo de `set_abnormal_data` usava `step` como fim, enquanto os outros três o usam como duração. Com início 200 e `step=13`, a fatia `[200:13]` fica vazia; com início 150 e `step=200`, entram 50 linhas em vez de 200. O teste original tinha 676 anomalias em vez de 826. A correção usa fim = início + duração e aceita início zero. A montagem por classe do teste original também foi trocada por blocos cronológicos, numa mudança registrada à parte.

**Conclusão da Parte II.** O Hitachi do tutorial foi integrado ao mesmo fluxo do GANF: `get_dataloader`, `MSELoss`, Adam, `UnsupervisedTrainer` e `StandardPredictor`, com o adaptador `HitachiScore`. No teste cronológico, F1 médio de 0,993 (desvio 0,003 entre cinco seeds), recall 1,0 e ROC-AUC perto de 1. O baseline sem treino teve F1 0,991. As anomalias sintéticas são grandes demais para diferenciar os métodos.

### Referências
- Repositório do tutorial: [{REPO.removeprefix('https://')}]({REPO}), commit `{COMMIT}`.
- Exemplo do GANF: `examples/MTSA.ipynb` do mesmo repositório.
- Wu, Park e Tirthapura. Graph-augmented normalizing flows for anomaly detection of multiple time series. AAAI, 2022. DOI 10.1609/aaai.v36i8.20829 (citado no código do GANF).
- Material da aula de Ciência de Dados do COMPET Moura, 12/10/2026.'''))

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        'colab': {'name': NOME, 'provenance': [], 'toc_visible': True},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'atividade4': {'repository': REPO, 'commit': COMMIT, 'csv_sha256': csv_sha, 'sourceHashes': hashes},
    }
    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        print('Notebook gerado:', target)
    return notebook


if __name__ == '__main__':
    build()
