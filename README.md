# NLP Intent Classifier for Gym Queries

This project is a simple NLP-based intent classifier designed to identify user intent in gym-related queries. It can be used as the core component of a gym chatbot or virtual assistant.

## Supported Intents

The classifier currently recognizes the following intents:
- Greeting
- Farewell
- Location
- Pricing

## How It Works

The system uses a supervised text classification pipeline.  
It is trained on example gym-related user queries that are manually annotated with intent labels.

The project is structured around two main files:
- **Training data file** — contains example user inputs and their corresponding intent labels
- **Classifier script** — trains a text categorizer using the annotated data and predicts intent for new inputs

## Current Limitations

- Only four intents are supported
- Limited training data
- Not yet optimized for production use

## Future Improvements

- Add more gym-related intents (e.g. memberships, class schedules, opening hours)
- Expand and clean the training dataset
- Integrate with a chatbot or messaging platform

## Use Case

This project serves as a foundation for a gym assistant capable of understanding user intent and routing queries to the appropriate response logic.
