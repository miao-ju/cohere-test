import enum

import cohere
from absl import app, flags
from cohere.responses.classify import Example

from . import credentials

FLAGS = flags.FLAGS

_QUESTION = flags.DEFINE_list('question', None, 'Question to ask...')


class eSupportOption(str, enum.Enum):
    ACCOUNT = 'Change account settings'
    SUBSCRIPTION = 'Manage subscription service'
    INFORMATION = 'Show company information'
    HUMAN_CONTACT = 'Chat with a human'


def main(_):
    co = cohere.Client(credentials.COHERE_API_KEY)

    examples = [
        Example('How can I update my password?', eSupportOption.ACCOUNT),
        Example('I would like to change my shipping address.', eSupportOption.ACCOUNT),
        Example('Can I request all of the information you have on me?', eSupportOption.ACCOUNT),
        Example('How can I create an account for the website?', eSupportOption.ACCOUNT),
        # ,
        Example('Can I cancel my subscription?', eSupportOption.SUBSCRIPTION),
        Example('How can I change my monthly subscription?', eSupportOption.SUBSCRIPTION),
        Example('Is there an option to change the frequency of my deliveries?', eSupportOption.SUBSCRIPTION),
        Example('Can I get coffee delivered every month?', eSupportOption.SUBSCRIPTION),
        # ,
        Example('How long has this company existed?', eSupportOption.INFORMATION),
        Example('How quickly should I consume my coffee once it has arrived?', eSupportOption.INFORMATION),
        Example('Where do you source your coffee from?', eSupportOption.INFORMATION),
        Example('Do you sell seasonal varieties of coffee?', eSupportOption.INFORMATION),
        # ,
        Example('How could I contact a human here?', eSupportOption.HUMAN_CONTACT),
        Example('None of your support options are what I need.', eSupportOption.HUMAN_CONTACT),
        Example('Is there an email address or phone number for the company?', eSupportOption.HUMAN_CONTACT),
        Example('My order never arrived.', eSupportOption.HUMAN_CONTACT),
    ]

    questions_to_classify = ['Can I update my monthly payment method?', 'Do you sell natural coffee?']

    response = co.classify(
        inputs=questions_to_classify,
        examples=examples,
    )

    print(response)
    for r, input in zip(response, questions_to_classify, strict=True):
        print(f'Question asked: {input}')
        print(f'\tPrediction: {r.predictions}')
        print(f'\tConfidence: {r.confidences}')

        print('\tLabels:')
        for k, v in r.labels.items():
            print(f'\t\t{k}={v}')

        print()


if __name__ == '__main__':
    app.run(main)
