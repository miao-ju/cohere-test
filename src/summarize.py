import cohere
from absl import app, flags

from . import credentials
from .utils import flags as ch_flags

FLAGS = flags.FLAGS

_DOC = ch_flags.DEFINE_path('doc', None, 'Document to summarise...')


def main(_):
    co = cohere.Client(credentials.COHERE_API_KEY)

    with open(FLAGS.doc, 'r') as f:
        text = f.read()

    response = co.summarize(text=text)
    print(response)


if __name__ == '__main__':
    app.run(main)
