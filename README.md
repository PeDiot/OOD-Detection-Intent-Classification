# OOD Detection for Intent Classification 

The methodology is inspired by [[1]](#1) and [[2]](#2) and the code is based on the [Todd](https://github.com/icannos/Todd) library and [ToddBenchmark](https://github.com/icannos/ToddBenchmark) framework implemented by [[3]](#3).

## TODO

- Add DistilBERT results
- Add title to PR/ROC curves
- Compute AUROC/AUPR wrt different layer selection for Maha & Cosine 
- Set cutoffs from scorer distributions (80% of train)

## References

### Articles

<a id="1">[1]</a> 
Pierre Colombo, Eduardo D. C. Gomes, Guillaume Staerman, Nathan Noiry, and Pablo Piantanida. 2022. <em><span
class="nocase">Beyond Mahalanobis-Based Scores for Textual OOD Detection.</span></em>

<a id="2">[2]</a> 
Wenxuan Zhou, Fangyu Liu, and Muhao Chen. 2021. <em><span
class="nocase">Contrastive Out-of-Distribution Detection for Pretrained Transformers.</span></em>

### Code

<a id="3">[3]</a> 
Darrin, Maxime, Manuel Faysse, Guillaume Staerman, Marine Picot, Eduardo
Dadalto Camara Gomez, and Pierre Colombo. 2023. <em><span
class="nocase">Todd: A tool for text OOD detection.</span></em> (version
0.0.1).

### Datasets

- [banking77](https://huggingface.co/datasets/banking77)
- [ATIS Airline Travel Information System](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem) 
- [Bitext - Customer Service Tagged Training Dataset for Intent Detection](https://github.com/bitext/customer-support-intent-detection-training-dataset)

### Model

- [philschmid/BERT-Banking77](https://huggingface.co/philschmid/BERT-Banking77) 