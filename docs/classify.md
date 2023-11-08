---
hide:
    - toc
    - path

title: Classification
---

# Classification

Cohere provides a simple end-point for <a href="https://docs.cohere.com/reference/classify">@classification</a>. This provides the ability to make predictions about which label best fits the specified text inputs. On this page, we will mock-up a basic customer-support classification tool for an e-commerce platform which sells coffee, and coffee equipment.

## The Task

We imagine that we are running a business where we sell coffee, and coffee-related products online. Our website is doing well and the number of customer queries we have to deal with is becoming unmanageable. Instead of hiring an external team to deal with this, we can provide common solutions to the user automatically.

First, we define a few different classes in which the queries may lie:

```python
import cohere
import enum


class eSupportOption(str, enum.Enum):
  ACCOUNT = 'Change account settings'
  SUBSCRIPTION = 'Manage subscription service'
  INFORMATION = 'Show company information'
  HUMAN_CONTACT = 'Chat with a human'
```

note, we have also included an option to talk to a human if the user requests it. Next, we define a few examples which can be used in our classification. These could be based on past interactions with customers which make these particularly easy to generate. We could provide something like:

```python
from cohere.responses.classify import Example

examples = [
  Example('How can I update my password?', eSupportOption.ACCOUNT),
  Example('I would like to change my shipping address.', eSupportOption.ACCOUNT),
  Example('Can I request all of the information you have on me?', eSupportOption.ACCOUNT),
  Example('How can I create an account for the website?', eSupportOption.ACCOUNT),
  #,
  Example('Can I cancel my subscription?', eSupportOption.SUBSCRIPTION),
  Example('How can I change my monthly subscription?', eSupportOption.SUBSCRIPTION),
  Example('Is there an option to change the frequency of my deliveries?', eSupportOption.SUBSCRIPTION),
  Example('Can I get coffee delivered every month?', eSupportOption.SUBSCRIPTION),
  #,
  Example('How long has this company existed?', eSupportOption.INFORMATION),
  Example('How quickly should I consume my coffee once it has arrived?', eSupportOption.INFORMATION),
  Example('Where do you source your coffee from?', eSupportOption.INFORMATION),
  Example('Do you sell seasonal varieties of coffee?', eSupportOption.INFORMATION),
  #,
  Example('How could I contact a human here?', eSupportOption.HUMAN_CONTACT),
  Example('None of your support options are what I need.', eSupportOption.HUMAN_CONTACT),
  Example('Is there an email address or phone number for the company?', eSupportOption.HUMAN_CONTACT),
  Example('My order never arrived.', eSupportOption.HUMAN_CONTACT),
]
```

All we have to do now is to send these examples, along with a question, to the <a href="https://docs.cohere.com/reference/classify">@classify</a> end-point, and we will receive an estimate for which class the question belongs to:

```python
import credentials
co = cohere.Client(credentials.COHERE_API_KEY)

questions_to_classify = ['Can I update my monthly payment method?', 'Do you sell natural coffee?']

response = co.classify(
  inputs=questions_to_classify,
  examples=examples,
)

```

... and with some manipulation of the structure of the response, we get something like:

```text
Question asked: Can I update my monthly payment method?
        Prediction: ['Manage subscription service']
        Confidence: [0.8488362]
        Labels:
                Change account settings=LabelPrediction(confidence=0.14907078)
                Chat with a human=LabelPrediction(confidence=0.0010854176)
                Manage subscription service=LabelPrediction(confidence=0.8488362)
                Show company information=LabelPrediction(confidence=0.0010076427)

Question asked: Do you sell natural coffee?
        Prediction: ['Show company information']
        Confidence: [0.99629194]
        Labels:
                Change account settings=LabelPrediction(confidence=0.0005747885)
                Chat with a human=LabelPrediction(confidence=0.001953109)
                Manage subscription service=LabelPrediction(confidence=0.0011801493)
                Show company information=LabelPrediction(confidence=0.99629194)
```
