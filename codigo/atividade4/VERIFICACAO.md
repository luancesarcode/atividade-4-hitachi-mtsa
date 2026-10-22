# Verificação da entrega — 21/10/2026

Escopo: correção tabular, adaptador Hitachi, experimento e artefatos da Atividade 4. O tutorial original de áudio/GANF não foi tratado como uma implementação validada por esta tarefa.

| Etapa | Resultado | Evidência |
|---|---|---|
| Compilação/sintaxe Python | PASS | `compileall` nos arquivos da atividade, sem erro |
| Tipos | PASS | Pyright nos dados, adaptador e experimento: zero erros e avisos |
| Lint | PASS | Ruff nos scripts novos/alterados da atividade |
| Testes | PASS | 13 verificações; 94% de cobertura nos dados/adaptador, adaptador 100% |
| Segredos | PASS | Busca focada sem ocorrências de chaves, senhas ou tokens |
| Revisão das mudanças | PASS | Fonte original preservada; correção do intervalo distinguida da mudança de protocolo |
| Notebook | PASS | Todas as seis células de código executadas em ordem, sem saída de erro; estrutura validada |
| PDF | PASS | Cinco páginas renderizadas e conferidas visualmente, sem texto cortado ou sobreposto |

A cobertura excluída corresponde ao carregador acústico, fora do experimento tabular escolhido. O teste do intervalo reproduz o erro silencioso, e o teste de integração passa por treinador e preditor reais do tutorial. Outro teste confere as métricas com uma matriz de confusão de contagens conhecidas.

O ZIP possui manifesto SHA-256 por arquivo e verificação de integridade. Resultados numéricos, predições e checkpoint são locais. Não houve push, publicação externa ou envio ao Classroom.

Limitação: o ambiente de execução local e o notebook foram verificados. A instalação a partir do zero no Google Colab não foi executada nesta tarefa.
