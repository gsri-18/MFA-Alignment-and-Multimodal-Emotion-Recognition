# Test accuracy of every model variant

Mean ± standard deviation over seeds (population standard deviation), in percent. Chance is 14.3% (7 balanced classes).

## Split `word` (420 test clips)

| pipeline | variant | accuracy | macro F1 | seeds |
|---|---|---|---|---|
| speech | mel | 99.8 ± 0.1 | 99.8 ± 0.1 | 3 |
| speech | wavlm | 99.8 ± 0.2 | 99.8 ± 0.2 | 3 |
| text | asr | 19.0 ± 0.3 | 12.5 ± 0.5 | 3 |
| text | gt | 14.3 ± 0.0 | 4.5 ± 1.3 | 3 |
| fusion | wavlm-asr-gated | 99.9 ± 0.1 | 99.9 ± 0.1 | 3 |
| fusion | wavlm-gt-concat | 99.9 ± 0.1 | 99.9 ± 0.1 | 3 |
| fusion | wavlm-gt-gated | 99.9 ± 0.1 | 99.9 ± 0.1 | 3 |

## Split `cross_speaker` (1400 test clips)

| pipeline | variant | accuracy | macro F1 | seeds |
|---|---|---|---|---|
| speech | mel | 52.2 ± 2.7 | 48.8 ± 5.1 | 3 |
| speech | wavlm | 65.6 ± 5.8 | 59.8 ± 7.1 | 3 |
| text | asr | 17.0 ± 0.7 | 13.4 ± 1.9 | 3 |
| text | gt | 14.3 ± 0.0 | 5.9 ± 1.7 | 3 |
| fusion | wavlm-asr-gated | 57.5 ± 11.2 | 55.4 ± 10.4 | 3 |
| fusion | wavlm-gt-concat | 55.3 ± 5.8 | 52.0 ± 4.4 | 3 |
| fusion | wavlm-gt-gated | 55.8 ± 1.6 | 50.4 ± 1.5 | 3 |
