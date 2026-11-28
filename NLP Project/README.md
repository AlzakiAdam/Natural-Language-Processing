# Review Sentiment Analysis under Domain Shift (NLP Project, Group 6)

**Option 2: Review Sentiment Analysis.** Ali Mohammed AlYousef (Team Leader), Adam Ibrahim Alzaki and Ahmed Murad Alshowaikhat.

**Question.** When trained on reviews from one domain, which method loses less macro F1 on the two unseen domains: TF-IDF with logistic regression, or pretrained sentence embeddings (all-MiniLM-L6-v2) with logistic regression?

**GitHub repository:** https://github.com/AlzakiAdam/Natural-Language-Processing/tree/main/NLP%20Project

## Run it
**Google Colab (recommended, CPU is enough)**
1. Upload this folder to Colab, or clone the repository in a Colab cell:
   ```
   !git clone https://github.com/AlzakiAdam/Natural-Language-Processing.git
   %cd "Natural-Language-Processing/NLP Project"
   ```
2. Open `G6_Sentiment_Domain_Shift.ipynb` and choose **Runtime -> Restart session and run all**.
3. Use the **Try it** cell: edit `REVIEW = "..."` and run the cell.

**Local Jupyter**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook G6_Sentiment_Domain_Shift.ipynb   # Kernel -> Restart & Run All
```
Requires Python 3.12+ (the submitted run used Python 3.12.13). A full run takes a few minutes on a CPU. The first run downloads the `all-MiniLM-L6-v2` weights (about 90 MB) from Hugging Face.

## Folder contents
| Path | Contents |
|---|---|
| `G6_Sentiment_Domain_Shift.ipynb` | Main notebook: Setup, Data, Method 1, Method 2, Results, Try it (sources are listed below) |
| `data/` | UCI Sentiment Labelled Sentences (Amazon, Yelp, IMDb). Downloaded automatically from UCI if missing |
| `results/` | All tables written by the notebook (CSV) |
| `figures/` | Figures used in the report and slides |
| `models/` | Saved final models used by the Try it cell (no retraining needed) |
| `requirements.txt` | Package versions used for the submitted run |

## Sources
**Data**
- D. Kotzias, *Sentiment Labelled Sentences*, UCI Machine Learning Repository, 2015, CC BY 4.0, doi:10.24432/C57604. https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- D. Kotzias, M. Denil, N. de Freitas, P. Smyth, "From group to individual labels using deep features," KDD 2015, doi:10.1145/2783258.2783380.

**Pretrained model**
- `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0), used frozen without fine-tuning. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- N. Reimers and I. Gurevych, "Sentence-BERT," EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
- W. Wang et al., "MiniLM," NeurIPS 2020. https://papers.nips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

**Libraries**
- scikit-learn (TF-IDF, logistic regression, metrics, splits): F. Pedregosa et al., JMLR 12, 2011. https://jmlr.org/papers/v12/pedregosa11a.html
- sentence-transformers and PyTorch (encoder), SciPy (McNemar test), pandas, NumPy, Matplotlib, joblib.

**Related work and methods**
- J. Blitzer, M. Dredze, F. Pereira, "Biographies, Bollywood, boom-boxes and blenders," ACL 2007. https://aclanthology.org/P07-1056/
- B. Pang, L. Lee, S. Vaithyanathan, "Thumbs up?," EMNLP 2002. https://aclanthology.org/W02-1011/
- D. Hendrycks et al., "Pretrained transformers improve out-of-distribution robustness," ACL 2020. https://aclanthology.org/2020.acl-main.244/
- Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," Psychometrika, 1947. https://doi.org/10.1007/BF02295996

**AI assistance**
With the instructor's permission, an AI assistant (Claude, Anthropic) was used to help draft and structure parts of the notebook and the report. The team reviewed, ran and can explain all code and results. This is also declared in the team appendix.

**Code**
Written by the team for this project, except for the libraries listed above (versions in `requirements.txt`).
