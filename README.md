# Deteção e quantificação de fugas por choque hidráulico

Código de análise de sinais de pressão transitória em condutas, desenvolvido no
âmbito de uma dissertação de mestrado. A partir do registo de pressão de uma
manobra de choque hidráulico, estima a **localização** de uma fuga e calcula
indicadores para a sua **quantificação**.

Os dados de ensaio e de simulação não fazem parte deste repositório — só o
código. A estrutura de pastas que os scripts esperam está descrita na
[secção 3](#3-onde-entram-os-dados).

---

## 1. O motor

| Ficheiro | O que faz |
| --- | --- |
| [`analysis.py`](analysis.py) | O núcleo. Leitura dos registos, deteção do arranque da manobra, celeridade, localização da fuga e indicadores de quantificação. Tudo o que é método está aqui, e nada mais o implementa. |
| [`pipe_configs/`](pipe_configs) | Geometria e propriedades de cada instalação, um ficheiro de texto por conduta. |
| [`validate_pipe_configs.py`](validate_pipe_configs.py) | Confirma que cada config é válida, sem correr uma análise. |

O motor é só cálculo: não desenha nada e não depende de nenhuma biblioteca de
gráficos.

## 2. O pipeline

Os *batches* correm o motor sobre todos os ensaios de uma instalação, emparelham
cada ensaio com fuga com a referência sem fuga da mesma manobra e caudal, e
escrevem uma linha por caso em `pair_single.csv`. Os relatórios só leem esse
ficheiro — não recalculam nada, para não poderem divergir do motor.

```
python batch_analysis.py            # bancada, conduta reta (CS)
python batch_analysis.py CC         # bancada, serpentina (CC)
python batch_analysis.py CS 250 350 # só estes caudais
python batch_simulated.py           # sinais do modelo numérico

python report_location.py CS        # localização: estimativas e erros
python report_indicators.py CS      # indicadores de quantificação
python report_repeatability.py CS   # dispersão entre repetições
python resultados.py                # todos os resultados, por metodologia
```

Os três relatórios aceitam `CS`, `CC` ou `SIM` e partilham a canalização comum
em [`report_common.py`](report_common.py).

## 3. Onde entram os dados

Os scripts descobrem os ensaios pelo **nome** do ficheiro, no formato
`KIND_FACILITY_MANEUVER_FLOW_(NL | LOCi_L<tamanho>)_[test]_DATE`. Basta colocar
as pastas na raiz deste repositório — estão todas em `.gitignore`:

```
Experimental_data_CS/   ensaios da conduta reta        (batch_analysis.py)
Experimental_data_CC/   ensaios da serpentina          (batch_analysis.py CC)
Numerical_data_CS/      sinais simulados, .dat         (batch_simulated.py)
```

## 4. Calibração e tabelas

| Ficheiro | O que faz |
| --- | --- |
| [`calibration_coeffs.py`](calibration_coeffs.py) | Calibra os indicadores de quantificação, `QL/Q0 = a * indicador`. Os coeficientes ajustam-se nos dados simulados e aplicam-se aos da bancada. |
| [`regressoes_sizing.py`](regressoes_sizing.py) | Compara formas alternativas de regressão para as três metodologias. |
| [`tabela_indicadores.py`](tabela_indicadores.py) | Um indicador por combinação, com o coeficiente de variação entre repetições. |
| [`tabela_validacao.py`](tabela_validacao.py) | O Excel de validação das equações de quantificação, uma folha por metodologia. |
| [`tabelas_apendice.py`](tabelas_apendice.py) | As tabelas dos apêndices, em CSV e num `.docx` formatado. |

## 5. Utilitários

| Ficheiro | O que faz |
| --- | --- |
| [`h0_ensaios.py`](h0_ensaios.py) | O regime permanente de cada ensaio de uma pasta: `H0`, perda de carga e `Q0`. |
| [`resample_frequency.py`](resample_frequency.py) | Reamostra um registo para outra frequência de aquisição. |
| [`valpal.py`](valpal.py) | As seis verificações de uma paleta de cores, incluindo simulação de daltonismo. |

## 6. Instalação

```
pip install -r requirements.txt
```

Python 3.10 ou superior. O `openpyxl` e o `python-docx` só são necessários para
os scripts que escrevem tabelas.

## 7. Licença

MIT — ver [LICENSE](LICENSE).
