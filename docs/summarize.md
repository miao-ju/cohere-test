---
hide:
    - toc
    - path

title: Summarize
---

# Summarizing Academic Papers

Cohere provides a <a href="https://docs.cohere.com/reference/summarize-2">@summarize</a> end-point. This provides the ability to summarize a given document, extracting just the key information. This has a large number of potential applications. One particular use-case I can think of, is summarizing academic papers to report on the key findings.

## The Task
On this page, we will explore whether the Cohere <a href="https://docs.cohere.com/reference/summarize-2">@summarize</a> API is capable of generating concise summaries of research papers. For the sake of this test, we will take the infamous paper: "Attention is all you need"[^1] and feed it to the summary model.

The code in this case is very simple. We simply need to read the document in as a string and send it straight to the API. To make this even easier, I created a simple text document from the PDF to avoid any unnecessary text extraction or parsing. We simply run the code:

```python
import cohere
from . import credentials


co = cohere.Client(credentials.COHERE_API_KEY)

# pass path to file as a command line argument
with open(FLAGS.doc, 'r') as f:
    text = f.read()

response = co.summarize(text=text)
```

This generates the output:

```text
The Transformer is a neural network architecture for sequence transduction that relies on repeated application of
attention mechanisms. By contrast, the dominant approach to sequence transduction, based on recurrent neural networks,
uses recurrence to propagate information through the network. We show that attention mechanisms can achieve better
performance with increased parallelism, making them well-suited for training large neural networks on hardware with many
parallel processors. Our Transformer model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task,
improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French
translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5
days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the
Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large
and limited training data.
```

While this does indeed provide a summary of the paper, it almost exactly mirrors the abstract. You could argue that the abstract provides the best summary of the paper -- but I think it would be nice to see a little more information contained within the paper in the summary.



[^1]:
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin., Attention is all you need., NeurIPS., 2017., <a href="https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html">https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html</a>
