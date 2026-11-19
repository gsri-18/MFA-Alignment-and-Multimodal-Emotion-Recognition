# Where fusion changes the speech only prediction

Same split and seed, clip by clip. fixed = speech wrong and fusion right, broken = speech right and fusion wrong.

| split | fusion variant | seed | speech acc | fusion acc | fixed | broken | gate mean |
|---|---|---|---|---|---|---|---|
| word | wavlm-asr-gated | 0 | 99.5 | 100.0 | 2 | 0 | 0.8836 |
| word | wavlm-asr-gated | 1 | 100.0 | 99.8 | 0 | 1 | 0.8858 |
| word | wavlm-asr-gated | 2 | 100.0 | 100.0 | 0 | 0 | 0.8343 |
| word | wavlm-gt-concat | 0 | 99.5 | 100.0 | 2 | 0 |  |
| word | wavlm-gt-concat | 1 | 100.0 | 100.0 | 0 | 0 |  |
| word | wavlm-gt-concat | 2 | 100.0 | 99.8 | 0 | 1 |  |
| word | wavlm-gt-gated | 0 | 99.5 | 100.0 | 2 | 0 | 0.8464 |
| word | wavlm-gt-gated | 1 | 100.0 | 100.0 | 0 | 0 | 0.972 |
| word | wavlm-gt-gated | 2 | 100.0 | 99.8 | 0 | 1 | 0.8726 |
| cross_speaker | wavlm-asr-gated | 0 | 73.0 | 44.1 | 80 | 484 | 0.8626 |
| cross_speaker | wavlm-asr-gated | 1 | 65.1 | 56.9 | 69 | 185 | 0.9285 |
| cross_speaker | wavlm-asr-gated | 2 | 58.7 | 71.6 | 258 | 77 | 0.8501 |
| cross_speaker | wavlm-gt-concat | 0 | 73.0 | 52.0 | 191 | 485 |  |
| cross_speaker | wavlm-gt-concat | 1 | 65.1 | 63.4 | 104 | 128 |  |
| cross_speaker | wavlm-gt-concat | 2 | 58.7 | 50.5 | 49 | 164 |  |
| cross_speaker | wavlm-gt-gated | 0 | 73.0 | 56.9 | 136 | 361 | 0.8384 |
| cross_speaker | wavlm-gt-gated | 1 | 65.1 | 53.6 | 36 | 198 | 0.8842 |
| cross_speaker | wavlm-gt-gated | 2 | 58.7 | 57.0 | 158 | 182 | 0.8958 |
