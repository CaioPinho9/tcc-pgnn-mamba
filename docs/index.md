# Modelos

`build_model(model_name, hyperparameters, device)` em [factory.py](../network/models/factory.py) instancia os cinco modelos usados no pipeline de treinamento. As configurações completas ficam em `config/models/`; os links abaixo apontam para os arquivos de arquitetura e para cada configuração.

## Interface tensorial

O pipeline fornece à rede `cnn_input` com formato `[B, C_s, N_t, N_r]`, em que `B` é o tamanho do lote, `C_s` é o número de fontes sísmicas, `N_t` é o número de amostras temporais e `N_r` é o número de receptores. `C_s` vem da configuração física da aquisição; `N_t` e `N_r` variam conforme o conjunto de dados.

Os modelos retornam um mapa de velocidade com formato `[B, 1, N_z, N_x]`, alinhado à grade do modelo-alvo. As configurações avaliadas definem uma saída de um canal e aplicam sigmoide. No treinamento, [train.py](../network/train.py) converte a saída normalizada para os limites de velocidade da configuração física antes da propagação direta.

## Blocos básicos

Nas arquiteturas convolucionais, `SubBlock` é a unidade básica: uma convolução 2D seguida de BatchNorm e LeakyReLU com inclinação 0,2. O kernel padrão é 3×3.

`DownBlock` aplica dois `SubBlock`s e depois `MaxPool2d(2)`. Ele retorna tanto o mapa reduzido quanto as características anteriores ao pooling, que podem alimentar uma conexão de salto.

`UpBlock` aplica dois `SubBlock`s e então aumenta a resolução por interpolação bilinear, por fator 2 ou até uma dimensão final explícita. Na U-Net, a concatenação com o mapa de salto acontece antes de chamar `UpBlock`.

As variantes com Mamba usam `UMambaBlock`, que aplica dois `SubBlock`s e uma camada `BiMambaLayer2D` opcional. `MambaDownBlock` acrescenta o pooling e retorna as características para possíveis conexões de salto; `MambaUpBlock` acrescenta o aumento bilinear de resolução. As implementações estão em [network/blocks](../../blocks/).

## Objetivo de treinamento compartilhado

O modelo estima a velocidade a partir dos sismogramas de entrada. O operador físico diferenciável propaga essa estimativa e produz sismogramas sintéticos. Depois da filtragem e normalização sísmica, a perda L1 compara os traços sintéticos aos observados e conduz o gradiente. O erro entre a velocidade prevista e a velocidade de referência é uma métrica de monitoramento, não o objetivo usado pelo otimizador. Esse fluxo está implementado em [pipeline.py](../network/pipeline.py) e [train.py](../network/train.py).

## Modelos e configurações

| Identificador | Página | Implementação | Configuração |
| --- | --- | --- | --- |
| `pgnn_mardan_2024` | [PGNN de Mardan baseada em Dhara (2022)](pgnn_mardan_2024.md) | [networks.py](../network/models/pinn_fwi/networks.py) | [pgnn_mardan_2024_config.json](../config/models/pgnn_mardan_2024_config.json) |
| `pgann_mardan_2024` | [PGANN de Mardan](pgann_mardan_2024.md) | [pgann_mardan_2024.py](../network/models/pgann_mardan_2024.py) | [pgann_mardan_2024_config.json](../config/models/pgann_mardan_2024_config.json) |
| `unet` | [U-Net](unet.md) | [unet.py](../network/models/unet.py) | [unet_config.json](../config/models/unet_config.json) |
| `mamba` | [Mamba](mamba_e_unetmamba.md) | [mamba.py](../network/models/mamba.py) | [mamba_config.json](../config/models/mamba_config.json) |
| `unetmamba` | [U-Net Mamba](mamba_e_unetmamba.md) | [mamba.py](../network/models/mamba.py) | [unetmamba_config.json](../config/models/unetmamba_config.json) |

## Figuras

Os diagramas de arquitetura e os mapas e gráficos de resultados ficam em pastas separadas. Consulte o [índice das figuras](figures/README.md) para ver a organização e a origem dos arquivos.