# Alignment summary

| clip | speaker | duration (s) | words |
|---|---|---|---|
| F2BJRLP1 | bu_f2b | 25.309 | 73 |
| F2BJRLP2 | bu_f2b | 28.647 | 74 |
| F2BJRLP3 | bu_f2b | 30.707 | 82 |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | isle_sess0131 | 4.125 | 5 |
| ISLE_SESS0131_BLOCKD02_02_sprt1 | isle_sess0131 | 3.875 | 5 |
| ISLE_SESS0131_BLOCKD02_03_sprt1 | isle_sess0131 | 4.5 | 5 |

## Checks per run

| run | clip | words match | phones | 30 ms or less | over 250 ms | inner pauses (s) | phones/s | speech log likelihood |
|---|---|---|---|---|---|---|---|---|
| arpa | F2BJRLP1 | yes | 308 | 30 | 0 | 12 (2.19) | 13.82 | -46.908 |
| arpa | F2BJRLP2 | yes | 345 | 26 | 0 | 8 (2.65) | 13.38 | -46.606 |
| arpa | F2BJRLP3 | yes | 372 | 26 | 1 | 11 (2.62) | 13.51 | -47.435 |
| arpa | ISLE_SESS0131_BLOCKD02_01_sprt1 | yes | 13 | 0 | 0 | 1 (0.15) | 7.83 | -49.365 |
| arpa | ISLE_SESS0131_BLOCKD02_02_sprt1 | yes | 12 | 0 | 1 | 1 (0.03) | 7.64 | -47.213 |
| arpa | ISLE_SESS0131_BLOCKD02_03_sprt1 | yes | 13 | 0 | 0 | 1 (0.03) | 7.26 | -49.093 |
| g2p_en | F2BJRLP1 | yes | 308 | 28 | 0 | 11 (2.16) | 13.81 | -46.866 |
| g2p_en | F2BJRLP2 | yes | 345 | 25 | 0 | 8 (2.65) | 13.38 | -46.593 |
| g2p_en | F2BJRLP3 | yes | 372 | 26 | 1 | 11 (2.62) | 13.51 | -47.396 |
| g2p_en | ISLE_SESS0131_BLOCKD02_01_sprt1 | yes | 13 | 0 | 0 | 1 (0.15) | 7.83 | -49.377 |
| g2p_en | ISLE_SESS0131_BLOCKD02_02_sprt1 | yes | 12 | 0 | 1 | 1 (0.03) | 7.64 | -47.139 |
| g2p_en | ISLE_SESS0131_BLOCKD02_03_sprt1 | yes | 13 | 0 | 0 | 1 (0.03) | 7.26 | -49.084 |
| mfa_ipa | F2BJRLP1 | yes | 300 | 31 | 1 | 10 (2.05) | 13.37 | -43.966 |
| mfa_ipa | F2BJRLP2 | yes | 340 | 35 | 2 | 7 (2.25) | 12.97 | -43.256 |
| mfa_ipa | F2BJRLP3 | yes | 368 | 40 | 1 | 12 (2.78) | 13.45 | -44.549 |
| mfa_ipa | ISLE_SESS0131_BLOCKD02_01_sprt1 | yes | 13 | 0 | 2 | 0 (0) | 3.52 | -45.577 |
| mfa_ipa | ISLE_SESS0131_BLOCKD02_02_sprt1 | yes | 11 | 0 | 1 | 0 (0) | 6.71 | -46.576 |
| mfa_ipa | ISLE_SESS0131_BLOCKD02_03_sprt1 | yes | 13 | 0 | 1 | 0 (0) | 3.03 | -45.574 |
| arpa_adapted | F2BJRLP1 | yes | 308 | 30 | 0 | 12 (2.19) | 13.82 | -46.085 |
| arpa_adapted | F2BJRLP2 | yes | 345 | 26 | 0 | 8 (2.65) | 13.38 | -45.773 |
| arpa_adapted | F2BJRLP3 | yes | 372 | 26 | 1 | 11 (2.62) | 13.52 | -46.547 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_01_sprt1 | yes | 13 | 0 | 0 | 1 (0.15) | 7.83 | -48.176 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_02_sprt1 | yes | 12 | 0 | 1 | 1 (0.03) | 7.64 | -46.143 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_03_sprt1 | yes | 13 | 0 | 0 | 1 (0.03) | 7.26 | -47.917 |

## Boundary difference from the arpa run (ms)

| run | clip | word mean | word median | word max | word > 50 ms (%) | phone mean | phone > 20 ms (%) |
|---|---|---|---|---|---|---|---|
| g2p_en | F2BJRLP1 | 0.5 | 0.0 | 30.0 | 0.0 | 0.5 | 0.5 |
| g2p_en | F2BJRLP2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.3 |
| g2p_en | F2BJRLP3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.3 |
| g2p_en | ISLE_SESS0131_BLOCKD02_01_sprt1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| g2p_en | ISLE_SESS0131_BLOCKD02_02_sprt1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 |
| g2p_en | ISLE_SESS0131_BLOCKD02_03_sprt1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mfa_ipa | F2BJRLP1 | 7.8 | 10.0 | 110.0 | 1.4 | different phone set | different phone set |
| mfa_ipa | F2BJRLP2 | 8.2 | 0.0 | 250.0 | 2.7 | different phone set | different phone set |
| mfa_ipa | F2BJRLP3 | 8.8 | 10.0 | 120.0 | 3.0 | different phone set | different phone set |
| mfa_ipa | ISLE_SESS0131_BLOCKD02_01_sprt1 | 231.5 | 20.0 | 1875.0 | 40.0 | different phone set | different phone set |
| mfa_ipa | ISLE_SESS0131_BLOCKD02_02_sprt1 | 35.0 | 10.0 | 160.0 | 20.0 | different phone set | different phone set |
| mfa_ipa | ISLE_SESS0131_BLOCKD02_03_sprt1 | 258.0 | 10.0 | 2460.0 | 10.0 | different phone set | different phone set |
| arpa_adapted | F2BJRLP1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| arpa_adapted | F2BJRLP2 | 0.1 | 0.0 | 10.0 | 0.0 | 0.2 | 0.3 |
| arpa_adapted | F2BJRLP3 | 0.1 | 0.0 | 10.0 | 0.0 | 0.1 | 0.0 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_01_sprt1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_02_sprt1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| arpa_adapted | ISLE_SESS0131_BLOCKD02_03_sprt1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
