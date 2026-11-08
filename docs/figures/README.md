# Figuras

- `models/` contém diagramas das arquiteturas avaliadas e seus blocos.
- `results/` mantém os mapas de velocidade das amostras 0 e 1 das quatro bases e os três gráficos métricos usados no capítulo de resultados do TCC: MAE de Marmousi e MSE de FlatFault_A e CurveFault_B. Gráficos de latência e pico de memória GPU não fazem parte desta seleção.

Os mapas da amostra 0 e os três gráficos métricos são as figuras referenciadas no capítulo; os mapas da amostra 1 ficam como resultados complementares. As tabelas, métricas e demais saídas da avaliação ficam em `results/evaluation/<versão>/` e podem ser regeneradas pelo notebook `02_evaluate_models.ipynb`.
