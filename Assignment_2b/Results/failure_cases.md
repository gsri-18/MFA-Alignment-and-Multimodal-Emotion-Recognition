# Failure cases

The most confident errors of the multimodal model (fusion_wavlm-gt-gated, seed 0), one per confusion pair. The word split is searched first, and these come from the cross_speaker split.

| # | split | clip | true | fusion predicts (confidence) | speech only | text only | fusion with ASR text | Whisper transcript |
|---|---|---|---|---|---|---|---|---|
| 1 | cross_speaker | YAF_tire_angry | angry | disgust (0.97) | angry | happy | disgust | Say the word tire. |
| 2 | cross_speaker | YAF_yearn_pleasant_surprise | pleasant_surprise | happy (0.96) | pleasant_surprise | disgust | happy | Say the word yearn. |
| 3 | cross_speaker | YAF_bean_angry | angry | sad (0.96) | fear | disgust | angry | Say the word bean. |
| 4 | cross_speaker | YAF_bath_fear | fear | neutral (0.96) | fear | happy | neutral | Say the word, Beth. |
| 5 | cross_speaker | YAF_jail_disgust | disgust | pleasant_surprise (0.96) | disgust | disgust | pleasant_surprise | Say the word jail. |
