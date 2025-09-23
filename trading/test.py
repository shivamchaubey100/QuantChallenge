import json

file = r"/Users/shivam/Documents/QuantChallenge/trading/example-game.json"
with open(file, 'r') as f:
    example_game = json.load(f)

strat = Strategy()
for event in example_game:
    strat.on_game_event_update(**event)