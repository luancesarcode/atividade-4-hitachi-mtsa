# Atividade 4: dados tabulares corrigidos e Hitachi no MTSA

Trabalho de Luan César de Souza Nascimento, estudante de Engenharia de Controle e Automação na UFPE, para a disciplina de ciência de dados do COMPET Moura.

O professor propôs duas tarefas sobre o tutorial [mtsa-tutorial-compet](https://github.com/CristoferSilva/mtsa-tutorial-compet) (commit `a5959467eca45f930341783a7006affcae5e6e88`):

1. achar o problema em `get_tabular_sample_data()`;
2. integrar o modelo Hitachi ao framework de aprendizado, como já existe para o GANF, e medir o desempenho em um conjunto de dados.

Os dados são o CSV sintético que vem no tutorial (5.000 linhas, 6 variáveis). Não são dados da Moura.

## O que tem aqui

| Arquivo | Conteúdo |
|---|---|
| `Luan_Cesar_de_Souza_Nascimento_Atividade_4.ipynb` | notebook executado, com índice, explicações e gráficos |
| `Apresentacao.pdf` | slides curtos com a parte mais desafiadora e a conclusão de cada parte |
| `codigo/` | módulos do tutorial, a correção, o adaptador do Hitachi, o experimento e os testes |

## Parte I: o erro

A função `set_abnormal_data` soma um deslocamento a quatro intervalos de linhas. Três deles usam `step` como duração, como em `(2*step, 3*step)`. O primeiro usava `step` como fim: `(start_point or step, step)`.

Com início 200 e `step=13`, a fatia fica `[200:13]`, vazia. O código roda sem avisar e nenhuma linha é alterada. Com início 150 e `step=200`, entram 50 linhas em vez de 200. Medindo sobre o CSV, o teste original tinha 676 linhas anômalas, e com a correção passa a ter 826.

A correção usa fim igual a início mais duração, aceita início zero e dá erro quando o intervalo é vazio ou sai dos dados. Também tirei um `np.random.seed(42)` que mexia no gerador global sem necessidade e deixei o caminho do CSV relativo ao arquivo.

O teste original também era montado por classe, com normais e anômalos colados. Isso não vaza linhas entre treino e teste, mas quebra a ordem no tempo: são 11 emendas, e 121 de 1.641 janelas de 12 linhas cruzam alguma delas. Troquei por blocos em ordem cronológica. É uma mudança de protocolo, separada da correção do intervalo.

## Parte II: Hitachi no framework

A classe `Hitachi` já existia no tutorial e não foi alterada. O que fiz foi ligá-la ao mesmo fluxo do exemplo do GANF:

| Etapa | Componente |
|---|---|
| Modelo | `Hitachi(input_dim=6)`, com 6 → 64 → 64 → 8 → 64 → 64 → 6 |
| Erro | `MSELoss`, com entrada igual ao alvo |
| Otimizador | Adam, taxa 0,001 |
| Treino | `UnsupervisedTrainer`, 30 épocas, só dados normais |
| Predição | `StandardPredictor` sobre o `HitachiScore` |

O `HitachiScore` (`codigo/mtsa/predictors/hitachi_score.py`) é a peça nova. O GANF já devolve um escore, e o Hitachi devolve a reconstrução, então o adaptador calcula o erro quadrático médio por linha. Escore maior significa mais anômalo.

Protocolo: treino nas linhas 0 a 2499, validação em 2500 a 2999 (só para o limiar, no percentil 99 dos escores normais) e teste em 3000 a 4999, com 1.174 linhas normais e 826 anômalas. A padronização usa só o treino. Cinco seeds, definidas antes de olhar o teste: 11, 22, 33, 42 e 55.

## Resultados

| Métrica | Hitachi (média ± desvio entre seeds) | Regra simples sem treino |
|---|---:|---:|
| ROC-AUC | 0,999996 ± 0,000007 | 1,000000 |
| Average precision | 0,999994 ± 0,000010 | 1,000000 |
| F1 | 0,993272 ± 0,002746 | 0,991002 |
| Recall | 1,000000 ± 0 | 1,000000 |
| Taxa de falso alarme | 0,009540 ± 0,003922 | 0,012777 |

A regra simples é a média dos quadrados das variáveis padronizadas, com o mesmo limiar. Ela acerta quase tudo também, e a diferença de F1 vem de alguns falsos alarmes (7 a 18 nas seeds do Hitachi, 15 na regra). Não dá para dizer que o Hitachi é melhor.

As notas são altas porque as anomalias são degraus constantes de 10 a 46 desvios-padrão. Qualquer método razoável separa isso.

## Como rodar

No Colab: abra o notebook e use Ambiente de execução → Executar tudo. Os módulos e o CSV estão embutidos nele, com SHA-256, então não precisa enviar nada. Eu rodei o notebook localmente, do início ao fim, em uma pasta vazia. No Colab ainda não conferi, e uma versão diferente do PyTorch pode mudar alguns falsos positivos.

Na máquina, com Python 3.12:

```bash
cd codigo
python -m pip install -r requirements-atividade4.txt
python -m pytest tests/test_atividade4.py -q
python -m atividade4.experimento
```

O `pytest` roda 13 testes (intervalos, divisão dos dados, direção do escore e a integração com treinador e preditor). O experimento regrava os resultados em `codigo/atividade4/resultados/`.

## Limitações

- Um único arquivo sintético e uma divisão temporal fixa. O resultado não diz nada sobre equipamento real.
- O GANF não foi reexecutado nesse protocolo, então não há comparação entre os dois modelos.
- Com seis entradas, a camada de 8 unidades não comprime nada.
- O desvio entre seeds mede só a variação do treino nesta mesma divisão, não a incerteza em dados novos.
- No `StandardPredictor` do tutorial, o caminho `pointwise=False` falha. Usei `pointwise=True`, que funciona, e não mexi nesse arquivo.

## Créditos

- O código em `codigo/mtsa/`, `codigo/examples/` (exceto `sample_data_utils.py`, que alterei) e o CSV vêm do tutorial de Cristofer Silva. O repositório original não declara licença, então os direitos desses arquivos são dele. A versão original da função está em `codigo/atividade4/fontes/`.
- O Hitachi e o GANF já faziam parte do tutorial. O GANF segue Wu, Park e Tirthapura, "Graph-augmented normalizing flows for anomaly detection of multiple time series", AAAI 2022 (DOI 10.1609/aaai.v36i8.20829).
- Fontes dos slides: Anton e Montserrat, licença OFL (arquivos em `codigo/atividade4/apresentacao/fonts/`).
