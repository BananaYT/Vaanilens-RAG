# Dataset attribution and source notes

## Hinglish & English Financial Scam/Fraud Text Dataset

- Dataset: `bolewara/hinglish-scam-text-dataset`
- Creator attributed by the dataset card: Anuj Bolewar
- Dataset card: https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset
- Raw data file (pinned revision `743799229923fc80fd7e9f2c764b5e9bd4ab3c9a`): https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset/resolve/743799229923fc80fd7e9f2c764b5e9bd4ab3c9a/data.json
- Declared licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Project use: downloaded at runtime by `prepare_scam_dataset.py`; the program writes local processed copies and source attribution metadata.

When redistributing the raw dataset or derivative data, retain attribution to the dataset creator, link the source and licence, identify modifications, and review the source repository's current licence/card before redistribution. This repository's importer does not imply that source labels have been independently verified.

## Important data-quality caveat

The dataset card describes 3,787 examples and labels `0` as benign and `1` as scam/fraudulent. It also says the English portion partly derives from the UCI SMS Spam Collection, while the Hinglish portion includes hand-written patterns. General spam may be promotional and not financial fraud. VaaniLens therefore preserves `source_label`, maps it to `scam_candidate` / `benign_candidate`, and marks it `source_label_unreviewed`. Do not treat every positive row as confirmed financial fraud or a real scam-call transcript.

## Official guidance sources used in `data/knowledge/scam_guidance.jsonl`

The project includes concise paraphrases with source URLs, rather than copied full documents:

- Reserve Bank of India, fraud-activity warning (29 August 2024): https://systemhealth.rbi.org.in/Scripts/FS_PressRelease.aspx_prid%3D58595%26fn%3D14.html
- RBI cautions on SMS/lottery offers: https://rbikehtahai.rbi.org.in/rbi-cautions-sms.html
- National Payments Corporation of India, fraud awareness: https://www.npci.org.in/fraud-awareness
- RBI digital-banking cyber security guidance: https://rbikehtahai.rbi.org.in/digital-banking-cyber-security.html
- RBI public guidance referencing cybercrime reporting: https://rbikehtahai.rbi.org.in/mm

The knowledge records are evidence-backed guidance, not labels for any individual call. Check the source links for current authoritative wording before relying on them operationally.
