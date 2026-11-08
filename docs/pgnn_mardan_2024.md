# PGNN de Mardan baseada em Dhara (2022)

Identificador da fábrica: `pgnn_mardan_2024`.

## Arquitetura implementada

A fábrica instancia `Autoencoder` em [networks.py](../network/models/pinn_fwi/networks.py). O codificador contém cinco blocos. Cada bloco aplica duas convoluções 3×3, cada uma seguida de BatchNorm e LeakyReLU com inclinação 0,2, e reduz a resolução com MaxPool2d de fator 2. O número de canais cresce de `2 C_s` até `32 C_s`.

O tensor codificado é achatado e projetado por uma camada totalmente conectada para um vetor de 392 elementos. O decodificador reorganiza esse vetor como `[1, 14, 28]` por amostra e aplica quatro blocos convolucionais com aumento bilinear de resolução. Os canais passam de 1 para 2, 4, 8 e 16. A última etapa ajusta a saída à grade de velocidade fornecida pelo pipeline; a convolução final 3×3 e a sigmoide geram um canal entre `m_min` e `m_max`.

O achatamento do codificador inclui a dimensão do lote. Por isso, a configuração associada a essa arquitetura fixa `model_batch_size` em 1.

## Parâmetros arquiteturais da configuração

| Parâmetro | Valor |
| --- | ---: |
| `n_blocks_encoder` | 5 |
| `n_blocks_decoder` | 4 |
| `final_size_encoder` | 392 |
| `decoder_initial_shape` | `[14, 28]` |
| `model_batch_size` | 1 |
| `model_output_min`, `model_output_max` | 0,0; 1,0 |
| `final_out_channels` | 1 |

Configuração completa: [pgnn_mardan_2024_config.json](../config/models/pgnn_mardan_2024_config.json). A implementação aplica à FWI a arquitetura PGNN associada a Dhara e Sen (2022), na formulação de Mardan (2024).

## Diagramas

A primeira imagem mostra a arquitetura PGNN de Dhara e Sen (2022), que serve de base para a formulação usada por Mardan (2024).

<img src="figures/models/pgnn_dhara2022_related_work.png" alt="Arquitetura PGNN de Dhara e Sen, base da formulação de Mardan" width="900">

O segundo diagrama detalha os blocos convolucionais usados pelo codificador e pelo decodificador. Ele não representa sozinho o autoencoder completo.

<img src="figures/models/pgnn_blocks.png" alt="Bloco convolucional, bloco do codificador e bloco do decodificador" width="850">