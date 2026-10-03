NBA Shot Make Probability Model

This project predicts the probability that an NBA shot will be made using player tracking data, shot characteristics, and historical shooter performance.

The final model is an XGBoost classifier with 21 input features. It achieved a validation log-loss of 0.6252115 and 0.6289117 when the fitting and validation seasons were reversed.

Data

The project uses the supplied datasets:

training.csv.gz: 425,719 shots with make/miss outcomes.

testing.csv.gz: 213,977 shots requiring predictions.

submission.csv: the provided prediction template.

The compressed datasets are read directly without extracting them. Shooter statistics are calculated from the supplied training data.

Model Features

Component

Features

Shooter ability

Shrunk shooter-zone make probability

Shot location

Shot zone, rim distance, court coordinates, and absolute lateral distance

Shot type

Jumper, layup, dunk, floater, post, or tip, including heaves

Defensive spacing

Closest defender’s distance at release

Defender movement

Closing rates over the preceding second and final quarter-second

Defender approach history

Distances at 1.0, 0.75, 0.50, and 0.25 seconds before release

Shooter movement

Shooter speed, dribble count, and a no-dribble indicator

Game context

Shot clock and game state

Contest information

Grouped contester count and log-transformed nearest-contester distance

XGBoost learns nonlinear relationships and interactions between these features, allowing defensive pressure and movement to affect different shot types differently.
