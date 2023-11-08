from pathlib import Path

import cohere
from absl import app, flags

from . import credentials
from .utils import flags as ch_flags

FLAGS = flags.FLAGS
_CV = ch_flags.DEFINE_path('cv', None, '.tex file to chat with')
_JOB_DESCRIPTION = ch_flags.DEFINE_path('job_description', None, 'text file to chat with')


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


def main(_):
    # jd_documents = produce_document('Job description', FLAGS.job_description)
    cv_documents = produce_document('CV', FLAGS.cv)

    all_documents = cv_documents
    # all_documents = [*jd_documents, *cv_documents]

    co = cohere.Client(credentials.COHERE_API_KEY)

    with open(FLAGS.job_description, 'r') as f:
        jd = f.read()

    preamble = """
        You will be provided with a job description. You will then be questioned about a candidate and whether they are
        suitable for the role. Please give an unbiased response, drawing on relevant items. Be positive. Here is the document:
    """

    preamble += jd

    preamble += 'The document has now ended... Please show specific instances in the job description where the candidate fulfills the brief. Do not try to be concise.'

    question_to_ask = """
        Please tell me a little about Miao. What experience does she have, and why would this make
        her a good product manager? Why would she be a good fit for this job inparticular? Please give specific reasoning.
    """

    response = co.chat(
        message=question_to_ask,
        documents=all_documents,
        preamble_override=preamble,
    )

    print(response.message)
    print(response.text)
    print()

    print('Looking at citations now:')
    print('===================================')

    for citation in response.citations:
        print(
            f'Used text: {citation["text"]} (idx=[{citation["start"]}, {citation["end"]}]) from documents {citation["document_ids"]}'
        )
        print('These document ids state the following:')

        for document_id in citation['document_ids']:
            # find corresponding document
            _document = next(filter(lambda x: x['id'] == document_id, response.documents))

            print(f'\tDocument ID: {document_id}')
            print(f'\tDocument title: {_document["title"]}')
            print(f'\tCitation: {_document["snippet"]}')

            print('\t################################')


if __name__ == '__main__':
    app.run(main)
