---
hide:
    - toc
    - path

title: RAG Chat
---

# RAG Chat

Cohere provides an end-point for chat, enhanced with retrieval-augmented generation (RAG): <a href="https://docs.cohere.com/reference/chat">@chat</a>. This provides the ability to talk with a (foundation-) language model, capable of providing grounded answers. Using RAG safeguards against hallucinations, a common downfall of LLMs; furthermore, it allows us to extract citations which give confidence in the response from the model.

## The Task

In this showcase, we will build a chat interface which has access to both a CV and a job description. We will then be able to talk with a language model to assess the suitablity of a candidate for a given role. Making use of retrieval-augmented generation will allow us to find specific references to the documents, preventing the language model from hallucinating.

As the focus here is to demonstrate the capabilities of RAG, I have not spent time on developing tools to parse text from PDF files; instead, I provide the CV and job description as simple text files (<a href="https://github.com/miao-ju/cohere-test/blob/main/docs/cv.md">CV</a>, <a href="https://github.com/miao-ju/cohere-test/blob/main/docs/jd.md">JD</a>).

The RAG implementation used by the Cohere API relies on snippets of information, each not exceeding c. 300 words in length. Breaking our documents down into snippets requires a little extra work, but ultimtately allows for more specific citations. Providing each of the snippets with an informative title ensures we can always trace the quoted citation back to the source. Given the simplicity of our raw data, we can use a simple function to produce a list of snippets from each document:

```python
from pathlib import Path


def produce_document(title: str, path: Path) -> list[dict[str, str]]:

    with open(path, 'r') as f:
        content = f.read()

    content = content.split('\n\n')

    documents = []
    for c in content:
        subtitle, *subcontents = c.split('\n')
        subcontents = '; '.join(subcontents)

        subtitle = subtitle.replace('# ', '')
        subcontents = subcontents.replace('- ', '')

        _doc = dict(title=f'{title}: {subtitle}', snippet=subcontents)
        documents.append(_doc)

    return documents
```

In the process of experimenting with the <a href="https://docs.cohere.com/reference/chat">@chat</a> API, it appears that using both the job descrition and CV as documents for RAG was counter productive. While outputs were fully cited, the model seemed to find it hard to distinguish between the documents. In order to avoid this, the CV is provided as a RAG document, and the job description is provided in the preamble. We define the preamble as

```python
# we pass the job description path as a command-line argument
with open(FLAGS.job_description, 'r') as f:
    jd = f.read()

preamble = f"""
    You will be provided with a job description. You will then be questioned about a candidate and whether they are
    suitable for the role. Please give an unbiased response, drawing on relevant items. Here is the document:
    {jd}
    The document has now ended... Please show specific instances in the job description where the candidate fulfills
    the brief. Do not try to be concise.
"""
```

Given the RAG documents, and the preamble, we are now ready to pose a question to the language model. We'll take this opportunity to ask about how my own experience compares with the job description!

```python
import cohere
import credentials


co = cohere.Client(credentials.COHERE_API_KEY)

question_to_ask = """
    Please tell me a little about Miao. What experience does she have, and why would this make
    her a good product manager? Why would she be a good fit for this job inparticular? Please give
    specific reasoning.
"""

# simple call to the API, providing our documents for RAG and custom preamble
response = co.chat(
    message=question_to_ask,
    documents=all_documents,
    preamble_override=preamble,
)
```

Upon receiving our response, we can take a look at what the language model says:

```text
Question:
    Please tell me a little about Miao. What experience does she have, and why would this make her a good product
    manager? Why would she be a good fit for this job inparticular? Please give specific reasoning.

Miao has extensive experience in technical product management. In her current role at StabilityAI, she led an
international team in the successful transition of Stable Animation from a research project to a market-ready product,
exclusively managing the product from launch to present. Miao routinely engaged with external customers and engineers to
ensure unfettered product access and collaborated with API partners, in conjunction with extensive market research, to
identify and address user stories, demonstrably enhancing the user experience. As a technical product manager, she has
also designed and documented product specifications across all scales and communicated these with both technical and non
technical stakeholders across the business. In addition, she has led efforts to substantially enhance the UI/UX for a
client-facing platform, resulting in a marked increase in user satisfaction, and has overseen the development of an
arithmetic recommendation platform.

Her prior experience at Tencent involved orchestrating the development of a feature that records users’ highlight
moments in the game Wild Rift. Miao led the conception of a tool capturing moments of interest within the game,
employing computer vision and analysis of the in-game log files and communicated with external partners to compile an
exhaustive set of requirements.

Overall, Miao has demonstrated her ability to successfully guide the work of research-inspired teams in building AI
models and products and make highly technical products simple and usable for customers.
```

Nice! This seems to have captured the spirit of my CV and the job description quite nicely. We can also take a look at the citations used to generate the output. Only a select few citations are shown below in the interest of space, but it is clear where the information is coming from:


```text
Used text:

    led an international team (idx=[103, 128]) from documents ['doc_0']

These document ids state the following:

        Document ID: doc_0
        Document title: CV: StabilityAI: Stable Animation
        Citation:
            Role: Technical Product Manager; Led an international team, successfully transitioning Stable Animation from
            a research project to a market-ready product as the exclusive product manager. The product is now
            responsible for over 10,000 generations per day.; – Routinely engaged with external customers and engineers
            to ensure unfettered product access from launch to present.; – Employed collaborations with API partners, in
            conjunction with extensive market research, to identify and address user stories; demonstrably enhancing
            user experience across the entire generation pipeline.

Used text:

    routinely engaged with external customers and engineers (idx=[291, 346]) from documents ['doc_0']

These document ids state the following:

        Document ID: doc_0
        Document title: CV: StabilityAI: Stable Animation
        Citation:
            Role: Technical Product Manager; Led an international team, successfully transitioning Stable Animation from
            a research project to a market-ready product as the exclusive product manager. The product is now
            responsible for over 10,000 generations per day.; – Routinely engaged with external customers and engineers
            to ensure unfettered product access from launch to present.; – Employed collaborations with API partners, in
            conjunction with extensive market research, to identify and address user stories; demonstrably enhancing
            user experience across the entire generation pipeline.

Used text:

    designed and documented product specifications (idx=[593, 639]) from documents ['doc_1']

These document ids state the following:

        Document ID: doc_1
        Document title: CV: StabilityAI: Fine-tuning
        Citation:
            Role: Technical Product Manager; – Designed and documented product specifications across all scales, from
            individual components to their integration.Communicated these with both technical and non-technical
            stakeholders across the business.; – Generated and optimised product roadmap through sustained: (i)
            collection and analysis of customer insights; (ii) benchmarking and evaluation of competitors products to
            identify target niches; and (iii) implementation of feedback loops with relevant shareholders to guarantee
            visibility and alignment.

Used text:

    communicated these with both technical and non-technical stakeholders (idx=[662, 731]) from documents ['doc_1']

These document ids state the following:

        Document ID: doc_1
        Document title: CV: StabilityAI: Fine-tuning
        Citation:
            Role: Technical Product Manager; – Designed and documented product specifications across all scales, from
            individual components to their integration.Communicated these with both technical and non-technical
            stakeholders across the business.; – Generated and optimised product roadmap through sustained: (i)
            collection and analysis of customer insights; (ii) benchmarking and evaluation of competitors products to
            identify target niches; and (iii) implementation of feedback loops with relevant shareholders to guarantee
            visibility and alignment.

Used text:

    led efforts to substantially enhance the UI/UX for a client-facing platform (idx=[774, 849]) from documents ['doc_2'

These document ids state the following:

        Document ID: doc_2
        Document title: CV: StabilityAI: DreamStudio
        Citation:
            Role: Technical Product Manager; Initiated and led efforts to substantially enhance the UI/UX for a client
            facing platform serving in excess of 600k users every month; ultimately, resulting in a marked increase in
            user satisfaction.; – Prioritized features in a constantly evolving backlog, liaising with stakeholders to
            mitigate unwarranted feature creep.

Used text:

    overseen the development of an arithmetic recommendation platform. (idx=[912, 978]) from documents ['doc_4']

These document ids state the following:

        Document ID: doc_4
        Document title: CV: ByteDance
        Citation:
            Role: Technical Product Manager; – Oversaw development of an arithmetic recommendation platform, currently
            serving the company’s education business.; – Prototyped, developed, and deployed two new features for a
            product providing interactive interpretation of mathematical problems. These features provided an increase
            in user engagement time of 20%.
```

By using RAG, we are able to have confidence in the outputs of the language model. It reduces the likelihood of hallucination significantly and is able to provide specific references to the given documents. Providing the job description as an additional document sounds appealing, but generations appeared to confuse the documents. This would result in the model providing 'facts' which appeared in one document or the other.
