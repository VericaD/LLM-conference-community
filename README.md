Can LLMs Dream of a Research Community

An end-to-end simulation of the ICLR peer-review process using open-weight LLMs.
Seven models each generate a set of papers; the papers are then reviewed by LLM
reviewers and accepted or rejected by an LLM area chair. The analysis asks whether
the pipeline reproduces known properties of human peer review, and whether models
show bias when reviewing their own output.

---

## Pipeline

```
frozen_ideas/  →  assembled_papers/  →  rag_runs/  →  review_outputs/
   ideas           generated papers      retrieval      reviews + decisions
```


