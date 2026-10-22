"""Treina e avalia o Hitachi com os componentes do MTSA do tutorial.

Executar da raiz: python -m atividade4.experimento
O notebook da entrega importa estas mesmas funções; CONFIG é o único lugar
com seeds, épocas, lote, taxa de aprendizado, divisão e regra do limiar.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from examples.sample_data_utils import get_tabular_sample_data
from mtsa.losses.mse_loss import MSELoss
from mtsa.models.hitachi import Hitachi
from mtsa.predictors.hitachi_score import HitachiScore
from mtsa.predictors.standard_predictor import StandardPredictor
from mtsa.trainers.unsupervised_trainer import UnsupervisedTrainer
from mtsa.utils.data import get_dataloader

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'atividade4' / 'resultados'

# Protocolo definido antes da avaliação; nenhum valor foi escolhido pelo teste.
CONFIG = {
    'seeds': [11, 22, 33, 42, 55],
    'epochs': 30,
    'batch_size': 64,
    'learning_rate': 0.001,
    'window_size': 1,
    'train_rows': 2500,            # linhas 0-2499 do bloco de desenvolvimento
    'threshold_quantile': 0.99,    # sobre escores normais da validação
    'saved_seed': 42,              # seed cujo modelo e predições são salvos
    'torch_threads': 2,
}
SEEDS = tuple(CONFIG['seeds'])
EPOCHS = CONFIG['epochs']


def loader(x, y, *, shuffle=False, config=CONFIG):
    # Autoencoder reconstrói cada vetor de seis variáveis: janela 1.
    return get_dataloader(np.asarray(x, dtype=np.float32), np.asarray(y),
                          window_size=config['window_size'], stride_size=1,
                          batch_size=config['batch_size'], shuffle=shuffle,
                          drop_last=False, num_workers=0)


def score(model, x, y, config=CONFIG):
    function = HitachiScore(model).eval()
    scores, labels = StandardPredictor().predict(
        function=function, dataloader=loader(x, y, config=config), pointwise=True,
        aggregation='mean', device='cpu')
    return scores.ravel(), labels.ravel().astype(int)


def metrics(y, scores, threshold):
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.,
            'precision': float(tp / (tp + fp)) if tp + fp else 0.,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.,
            'far': float(fp / (fp + tn)), 'mar': float(fn / (fn + tp)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'threshold': float(threshold)}


def audit_original():
    """Reproduz a primeira fatia original e compara seu efeito com a correção."""
    path = ROOT / 'atividade4/fontes/original_sample_data_utils.py'
    source = path.read_text(encoding='utf-8')
    if '(start_point or step, step)' not in source:
        raise ValueError('A fonte preservada não contém o intervalo original esperado.')
    start_point, step = 200, 13
    old_start, old_end = start_point or step, step
    old_y = np.zeros(1000)
    old_y[old_start:old_end] = 1
    from examples.sample_data_utils import set_abnormal_data
    _, new_y = set_abnormal_data(np.zeros((1000, 6)), 13, np.zeros(1000),
                                features=[0], start_point=200, magnitude=7)
    return {'original_first_interval': [200, 13], 'corrected_first_interval': [200, 213],
            'original_positive_rows_200_213': int(old_y[200:213].sum()),
            'corrected_positive_rows_200_213': int(new_y[200:213].sum()),
            'original_other_first_interval': [150, 200],
            'corrected_other_first_interval': [150, 350],
            'sine_is_constant': True,
            'note': 'magnitude * sin(pi/4) é constante: deslocamento, não senoide temporal.'}


def prepare_data(config=CONFIG):
    """Divide em treino/validação/teste e padroniza com estatísticas do treino."""
    xdev, ydev, xtest, ytest, metadata = get_tabular_sample_data(return_metadata=True)
    cut = config['train_rows']
    xfit, xval = xdev[:cut], xdev[cut:]
    scaler = StandardScaler().fit(xfit)
    fit, val, test = [np.asarray(scaler.transform(x), dtype=np.float32) for x in [xfit, xval, xtest]]
    return {'fit': fit, 'val': val, 'test': test, 'yfit': ydev[:cut], 'yval': ydev[cut:],
            'ytest': ytest, 'scaler': scaler, 'metadata': metadata}


def train_one_seed(seed, data, config=CONFIG):
    """Modelo, perda, otimizador, treinador e preditor do framework; limiar pela validação."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Hitachi(input_dim=data['fit'].shape[1]).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = UnsupervisedTrainer()
    started = time.perf_counter()
    trainer.fit(model=model, optimizer=optimizer, loss=MSELoss(),
                train_loader=loader(data['fit'], data['yfit'], shuffle=True, config=config),
                epochs=config['epochs'])
    elapsed = time.perf_counter() - started
    val_scores, _ = score(model, data['val'], data['yval'], config)
    threshold = float(np.quantile(val_scores, config['threshold_quantile']))
    test_scores, test_labels = score(model, data['test'], data['ytest'], config)
    record = {'model': 'Hitachi', 'seed': seed, **metrics(test_labels, test_scores, threshold),
              'train_seconds': elapsed, 'train_final_loss': float(trainer.min_error)}
    return record, model, val_scores, test_scores


def baseline(data, config=CONFIG):
    """Regra fixa: média dos quadrados das variáveis padronizadas pelo treino."""
    val_baseline = np.square(data['val']).mean(axis=1)
    test_baseline = np.square(data['test']).mean(axis=1)
    threshold = np.quantile(val_baseline, config['threshold_quantile'])
    return {'model': 'Distância ao normal', **metrics(data['ytest'], test_baseline, threshold)}, test_baseline


def summarize(records):
    return {key: {'mean': float(np.mean([r[key] for r in records])),
                  'sd': float(np.std([r[key] for r in records], ddof=1))}
            for key in ['roc_auc', 'average_precision', 'f1', 'precision', 'recall', 'far', 'mar']}


def run(config=CONFIG, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['torch_threads'])
    torch.use_deterministic_algorithms(True)
    data = prepare_data(config)
    fit, val, test, ytest, scaler, metadata = (data[k] for k in ['fit', 'val', 'test', 'ytest', 'scaler', 'metadata'])
    audit = audit_original()
    provenance = json.loads((ROOT/'atividade4/fontes/proveniencia.json').read_text(encoding='utf-8'))
    settings = {**config, 'threshold_rule': 'percentil 99 dos escores normais de validação',
                'train_rows': len(fit), 'validation_rows': len(val), 'test_rows': len(test),
                'test_normals': int((ytest == 0).sum()), 'test_anomalies': int(ytest.sum()),
                'architecture': '6 -> 64 -> 64 -> 8 -> 64 -> 64 -> 6; ReLU; saída linear',
                'device': 'cpu', 'dataset': 'sintético do tutorial; desvios constantes injetados',
                'selection': 'Sem busca de hiperparâmetros e sem escolha de seed pelo teste.'}
    records = []
    for seed in config['seeds']:
        record, model, _, test_scores = train_one_seed(seed, data, config)
        records.append(record)
        print(json.dumps(record), flush=True)
        if seed == config['saved_seed']:
            torch.save({'state_dict': model.state_dict(), 'input_dim': 6, 'seed': seed,
                        'threshold': record['threshold'], 'scaler_mean': np.asarray(scaler.mean_).tolist(),
                        'scaler_scale': np.asarray(scaler.scale_).tolist()}, output/'hitachi_seed42.pt')
            pd.DataFrame({'time': metadata['test_indices'], 'y_true': ytest,
                          'score_hitachi': test_scores}).to_csv(output/'predicoes_seed42.csv', index=False)

    base, _ = baseline(data, config)
    frame = pd.DataFrame(records)
    report = {'provenance': provenance, 'settings': settings, 'bug_evidence': audit,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'pandas': pd.__version__},
              'runs': records, 'summary': summarize(records), 'baseline': base,
              'limitations': ['Um único arquivo sintético e uma divisão temporal fixa.',
                              'Anomalias adicionadas têm amplitudes muito maiores que a variação normal.',
                              'Desvio entre seeds mede apenas inicialização/ordem de treino, não novos conjuntos.',
                              'Não demonstra desempenho em dados reais nem superioridade sobre GANF.',
                              'Hitachi já existia no tutorial; contribuição é a integração completa e a avaliação.',
                              'Com seis entradas, a camada de oito unidades não comprime: adapta-se a arquitetura existente.']}
    frame.to_csv(output/'metricas_seeds.csv', index=False)
    (output/'resultados.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'auditoria_particao.json').write_text(json.dumps(metadata), encoding='utf-8')
    print('RESULTADOS', output, flush=True)
    return report


if __name__ == '__main__':
    run()
