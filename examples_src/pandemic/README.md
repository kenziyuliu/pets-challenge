# Simple Example Model (Track B: Pandemic Forecasting)

This is an example solution that implements both federated and centralized versions of simple, single-parameter disease model for Track B: Pandemic Forecasting.

The model is based on the discrete-time [SIR model](https://people.wku.edu/lily.popova.zhuhadar/) of infectious disease dynamics in a population. This model's description of disease state closely matches how the disease state is represented in the simulation that produced the challenge dataset.

```math
I_{t+1} - I_t = \underbrace{\beta I_t \frac{S_t}{N}}_{\text{new infections}} - \underbrace{\gamma I_t \vphantom{\frac{S_t}{N}}}_{\text{recoveries}}
```

Because our task is forecasting the risk of becoming infected during the test period, we are primarily interested in the first term of the difference equation for the number of infections, which represents the number of _new_ infections. (The second term represents the number of infected individuals who recover.) The $\beta$ parameter in the first term represents the average number of contact infections made by an infected individual. We use a timestep $\Delta t$ of 7 days to match the duration of test period. If designate the number of new infections over the time step to be a dependent variable $y$, and $I_t S_t / N$ to be the dependent variable $x$, we have a simple linear functional equation of the form

```math
y_t = \beta x_t $$
```

We can fit an estimate $\hat \beta$ for our model using each day in the training period of the dataset as an observation (excluding the final week due to an incomplete lookahead window). There are many estimation methods that could be used—we will use a simple [ordinary least squares](https://en.wikipedia.org/wiki/Simple_linear_regression#Simple_linear_regression_without_the_intercept_term_(single_regressor)) estimator.

```math
\hat \beta = \frac{\sum_{t} x_t y_t}{\sum_{t} x_t^2} $$
```

To produce the target predictions, we use our estimated $\hat \beta$ parameter with the population disease state on day 56 to estimate the number of new infections over the next 7 days. We then make the naive prediction that all susceptible individuals are equally likely to become infected, so each individual's risk is the estimated new infections divided by the number of susceptible individuals.

### Federated version

For the federated version, we will simply treat each day from each partition as an observation. The sums in the above equation for $\hat \beta$ will be both across time $t$ and across partitions.

## Running this example

To run this example, first ensure that you have valid data files in `data/pandemic` and have the runtime image built or pulled.

Then, to pack up the example submission:

```bash
SUBMISSION_TRACK=pandemic make pack-example
```

and then to run the submission locally

```bash
# To federated submission
SUBMISSION_TYPE=federated SUBMISSION_TRACK=pandemic make test-submission

# To run centralized submission
SUBMISSION_TYPE=centralized SUBMISSION_TRACK=pandemic make test-submission
```
