TRAIN_DATA = [
    # Greeting
    ("hello", {"cats":{"greeting": 1.0, "farewell": 0.0, "location": 0.0, "pricing": 0.0}}),
    ("how are you", {"cats": {"greeting": 1.0, "farewell": 0.0, "location": 0.0, "pricing":0.0}}),

    # Farewell
    ("bye", {"cats":{"greeting": 0.0, "farewell": 1.0, "location": 0.0, "pricing": 0.0}}),
    ("goodbye", {"cats":{"greeting": 0.0, "farewell": 1.0, "location": 0.0, "pricing": 0.0 }}),

    # Location
    ("where are you located", {"cats":{"greeting": 0.0, "farewell": 0.0, "location": 1.0, "pricing": 0.0}}), 
    ("what is your adress", {"cats":{"greeting": 0.0, "farewell": 0.0, "location": 1.0, "pricing":0.0}}),

    # Pricing
    ("how much is a membership", {"cats": {"greeting":0.0, "farewell": 0.0, "location":0.0, "pricing":1.0}}),
    ("how much is a day pass", {"cats":{"greeting": 0.0, "farewell": 0.0, "location":0.0, "pricing": 1.0}})
]