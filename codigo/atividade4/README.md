# Atividade 4 — entrega de Luan César de Souza Nascimento

O desafio foi resolvido no tutorial público localizado em:
https://github.com/CristoferSilva/mtsa-tutorial-compet

Referência exata: commit `a5959467eca45f930341783a7006affcae5e6e88`.

## Arquivos

- `atividade4/experimento.py`: experimento em linha de comando; `CONFIG` reúne seeds, épocas, lote, taxa, divisão e limiar.
- `atividade4/criar_notebook.py`, `validar_notebook.py`, `gerar_apresentacao.py`, `montar_entrega.py`: geram o notebook, a apresentação e o ZIP em uma pasta `saida/` ao lado de `codigo/`.
- `atividade4/resultados/`: resultados medidos.
- `tests/test_atividade4.py`: verificações dos intervalos, dos dados e da integração do modelo.
- `atividade4/fontes/`: código original e proveniência dos dados.

## Parte I — problema e correção

O intervalo `(start_point or step, step)` gera uma fatia vazia na chamada real com início 200 e `step=13`: `[200:13]`. O erro é silencioso. A correção usa início explícito e fim `início + step`; início zero também é respeitado. Interpretar `step` como duração é consistente com os outros três intervalos do injetor. Na chamada com início 150 e `step=200`, o intervalo passa de 50 para 200 linhas.

A prova isolada compara zero linhas alteradas na primeira faixa original com 13 linhas após a correção. Ela está no notebook e em `resultados/resultados.json`.

O método original ainda montava teste como normais seguidos de anômalos, retirando a ordem temporal e criando junções artificiais. O modo `legacy` preserva essa montagem para comparação, com o injetor corrigido. O padrão `chronological`, usado na entrega, separa blocos contíguos antes de injetar anomalias somente no teste. Esta mudança de protocolo é distinta da correção mínima do intervalo.

O caminho do CSV agora é relativo ao módulo. O carregamento tabular não depende dos pacotes de áudio e não altera o gerador aleatório global.

## Parte II — implementação no contexto do framework

A arquitetura `Hitachi` já existia no tutorial. Reutilizamos essa implementação: 6 → 64 → 64 → 8 → 64 → 64 → 6, ReLU interna e saída linear. Não reivindicamos sua autoria. A integração completa usa `get_dataloader`, `MSELoss`, Adam, `UnsupervisedTrainer`, o novo adaptador `HitachiScore` e `StandardPredictor`. Maior MSE de reconstrução significa maior anomalia. Janela 1 corresponde ao autoencoder de vetores por instante.

O exemplo de GANF continua preservado em `examples/MTSA.ipynb`. Não reexecutamos GANF para afirmar superioridade.

## Protocolo e resultados

Dados: 5.000 linhas sintéticas e seis variáveis do CSV do tutorial. `U` e `time` ficam fora das entradas. A origem sintética e a injeção de anomalias ficam explícitas; não são resultados de manutenção da Moura.

- Treino: linhas 0–2499, normais; validação: 2500–2999, normais; teste: 3000–4999.
- Teste: 1.174 normais e 826 anômalas, preservadas em ordem temporal.
- Normalização ajustada apenas no treino; o mesmo ajuste é aplicado à validação e ao teste.
- 30 épocas, lote 64, Adam 0,001, CPU; seeds 11, 22, 33, 42 e 55 definidas previamente.
- Limiar: percentil 99 dos escores normais de validação. Nenhum ajuste pelo teste.

| Métrica | Hitachi: média ± desvio entre seeds | Baseline distância ao normal |
|---|---:|---:|
| ROC-AUC | 0,999996 ± 0,000007 | 1,000000 |
| Average precision | 0,999994 ± 0,000010 | 1,000000 |
| F1 | 0,993272 ± 0,002746 | 0,991002 |
| Recall | 1,000000 ± 0,000000 | 1,000000 |
| FAR | 0,009540 ± 0,003922 | 0,012777 |

Na seed 42, foram 826 verdadeiros positivos, 1.156 verdadeiros negativos, 18 falsos positivos e nenhuma anomalia perdida. Essa seed foi fixada antes da avaliação, não escolhida pelo melhor resultado.

O baseline é a média dos quadrados das seis variáveis padronizadas pelo treino; seu limiar também é calibrado no percentil 99 da validação. A pequena diferença de F1 não sustenta superioridade geral do Hitachi. Os desvios entre seeds medem apenas variação de treinamento nesta mesma divisão, não incerteza de novos conjuntos.

## Qual foi a parte mais desafiadora?

Garantir a coerência entre os dados e a avaliação: uma fatia vazia não interrompe a execução, e reagrupar linhas por classe pode destruir a sequência temporal. A integração também exigiu adaptar as dimensões dos tensores e converter reconstrução em um escore cujo sentido fosse consistente com os rótulos.

## Limitações

É um único arquivo sintético, em uma divisão temporal fixa, com desvios constantes muito maiores que a variação normal. Resultados altos aqui não comprovam eficácia em equipamentos reais. Não foi demonstrada superioridade sobre GANF ou sobre o baseline. Com seis variáveis, a camada de oito unidades não é um gargalo que comprime a entrada; a arquitetura original foi preservada para a integração pedida.

## Como repetir

Na raiz que contém `mtsa/`, use Python 3.12 e as dependências de `requirements-atividade4.txt`. O `requirements.txt` original foi preservado.

```powershell
python -m pip install -r requirements-atividade4.txt
python -m pytest tests/test_atividade4.py -q
python -m atividade4.experimento
python -m atividade4.criar_notebook
python -m atividade4.validar_notebook
python -m atividade4.gerar_apresentacao
python -m atividade4.montar_entrega
```

O notebook também roda sozinho no Colab: abra o `.ipynb` e use "Executar tudo". A execução local foi verificada; a do Colab, não. A geração do PDF usa o Chrome ou o Edge em modo headless.
