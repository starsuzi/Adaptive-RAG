# Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

Official Code Repository for the paper ["Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"](https://arxiv.org/pdf/2403.14403.pdf) (NAACL 2024).

## Abstract

<div align="center">
  <img alt="Adaptive-RAG Overview" src="./images/adaptiverag.png" width="800px">
</div>
Retrieval-Augmented Large Language Models (LLMs), which incorporate the non-parametric knowledge from external knowledge bases into LLMs, have emerged as a promising approach to enhancing response accuracy in several tasks, such as Question-Answering (QA). However, even though there are various approaches dealing with queries of different complexities, they either handle simple queries with unnecessary computational overhead or fail to adequately address complex multi-step queries; yet, not all user requests fall into only one of the simple or complex categories. In this work, we propose a novel adaptive QA framework that can dynamically select the most suitable strategy for (retrieval-augmented) LLMs from the simplest to the most sophisticated ones based on the query complexity. Also, this selection process is operationalized with a classifier, which is a smaller LM trained to predict the complexity level of incoming queries with automatically collected labels, obtained from actual predicted outcomes of models and inherent inductive biases in datasets. This approach offers a balanced strategy, seamlessly adapting between the iterative and single-step retrieval-augmented LLMs, as well as the no-retrieval methods, in response to a range of query complexities. We validate our model on a set of open-domain QA datasets, covering multiple query complexities, and show that ours enhances the overall efficiency and accuracy of QA systems, compared to relevant baselines including the adaptive retrieval approaches.

## Citation
If you found the provided code with our paper useful, we kindly request that you cite our work.
```BibTex
@inproceedings{jeong2024adaptiverag,
  author       = {Soyeong Jeong and
                  Jinheon Baek and
                  Sukmin Cho and
                  Sung Ju Hwang and
                  Jong Park},
  title        = {Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity},
  booktitle={NAACL},
  year={2024},
  url={https://arxiv.org/abs/2403.14403}
}
```
