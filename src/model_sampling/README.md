# Model Sampling

this directory contains implementations of various sampling methods.
 - relational_sampler.py: The learnable sampling implementation.
 - random_sampler.py: samples edge in each neighborhood randomly.
 - inverse_sampler.py: gives rarer edges a high probability of being sampled.
 - sigmoid_sampler.py: This was just experimented with as a possible improvement.

These are each attached onto a different model. They all use sampling_impl.py as the
the underlying implementation.
