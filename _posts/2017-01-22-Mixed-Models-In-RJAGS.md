I'm going to explain how to use RJAGS for a linear mixed effects model.

Note: This post assumes that you're already familiar with the concepts of linear mixed effects models and Bayesian inference. Additionally, some prior experience with using RJAGS to perform simple Bayesian linear regression will help, but it's not essential.

I'm going to use the `Teams` dataset from the `Lahman` package. We're going to model runs scored (`R`) vs hits (`H`) by teams (`teamID`). For the sake of simplicity, I've filtered the dataset to only include 4 teams since 2000.

Note: If you want to make an accurate mixed effects model, don't use only 4 teams. It's hard to assume that 4 teams are normally distributed around the mean (of their intercepts and their slopes), because the sample size is just too small. I'm only using 4 because the focus of this blog post is to explain *how to code* a mixed effects model, as opposed to explaining *what a mixed effects model is*.

``` r
# wrangle relevant dataset
data = Teams %>%
  filter(yearID >= 2000) %>%
  filter(teamID %in% c("ATL", "BAL", "BOS", "NYA")) %>%
  select(R, H, teamID)
```

Here's a scatterplot of Runs vs. Hits by team:

``` r
ggplot(data, aes(x = H, y = R, color = teamID)) +
  geom_point() +
  ggtitle("Runs v. Hits by Team")
```

![](https://amolmane1.github.io/images/2017-01-22-Mixed-Models-In-R-scatterplot.png)

As you can see, all teams have similar (but not the same) slopes and intercepts. This is great for a mixed effects model.

### Standard model

Before we fit a Bayesian model though, let's make a mixed effects model using the `lmer` function from the `lme4` package for comparison.

``` r
# fit model
model_lmer = lmer(R ~ I(scale(H)) + (1 + I(scale(H))|teamID),
             data = data)

# fixed effects
summary(model_lmer)
```

    ## Linear mixed model fit by REML ['lmerMod']
    ## Formula: R ~ I(scale(H)) + (1 + I(scale(H)) | teamID)
    ##    Data: data
    ##
    ## REML criterion at convergence: 643.1
    ##
    ## Scaled residuals:
    ##      Min       1Q   Median       3Q      Max
    ## -2.76741 -0.56128 -0.03797  0.66286  2.13252
    ##
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev. Corr
    ##  teamID   (Intercept) 1029.3   32.08        
    ##           I(scale(H))  260.3   16.14    0.81
    ##  Residual             1404.8   37.48        
    ## Number of obs: 64, groups:  teamID, 4
    ##
    ## Fixed effects:
    ##             Estimate Std. Error t value
    ## (Intercept)  782.259     16.772   46.64
    ## I(scale(H))   76.003      9.544    7.96
    ##
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## I(scale(H)) 0.656

As you can see, the fixed effects are:

    beta0 = 782.259
    beta1 = 76.003

And the random effects are:

``` r
# random effects by team
coef(model_lmer)$teamID
```

    ##     (Intercept) I(scale(H))
    ## ATL    777.3443    79.18597
    ## BAL    745.0969    54.87308
    ## BOS    786.7855    83.41029
    ## NYA    819.8106    86.54292

### RJAGS Model

And now we're ready to work on the RJAGS version of this model.

``` r
# set seed
model_inits <- list(.RNG.name="base::Super-Duper", .RNG.seed=1)
```

This might be familiar to you if you've used RJAGS before - it's RJAGS's way of specifying a seed for randomness

``` r
# prior fixed effects
zero = c(0, 0)
```

Above is the first novel piece of code needed for an RJAGS mixed effects model as opposed to a standard RJAGS linear model. `zero` is an array that will be used in the JAGS model to store the fixed effects.

``` r
# variance-covariance matrix of the random effects
R = matrix(c(.1, 0, 0, .1),
           nrow = 2,
           ncol = 2,
           byrow = TRUE)
```

R is a variance-covariance matrix of the random effects. In the model, it'll be used to help calculate the random effects by `teamID`.

``` r
# collect model data
model_data <- list( id = as.integer( factor(data$teamID) ),
                    x = I(scale(data$H))[ , 1],
                    y = data$R,
                    N = nrow(data),
                    F = 4,
                    zero = zero,
                    R = R)
```

Above is all the model data we pass into the model. - `id` is a numeric version of `teamID` (we need it to be numeric because the model's going to do calculations using it). - `x` is the `Hits` data for all the teams over all the years, scaled. - `y` is the `Runs` data. - `N` is the number of rows. - `F` is the number of teams. - I've explained `zero` and `R` earlier.

**Now** we can get to the meat of this blog post - the RJAGS model. If you're not familiar with RJAGS, the next code block will look intimidating, but hopefully I can help you understand it better. It's basically a coded version of the distributions of the priors and posteriors.

Note: Parts of this code have been directly taken from <http://bendixcarstensen.com/Bayes/Cph-2012/pracs.pdf>

``` r
model_string = "
model {
  # Define model for each observational unit
  for( j in 1:N ) {
    mu[j] <- ( beta[1] + u[id[j],1] ) + ( beta[2] + u[id[j],2] ) * x[j]
    y[j] ~ dnorm( mu[j], 1/(sigma.e^2) )
  }

  # Intercept and slope for each person, including random effects
  for( f in 1:F ) {
    u[f, 1:2] ~ dmnorm(zero, Omega.u)
  }

  # Priors:
  # Fixed intercept and slope
  beta[1] ~ dnorm(0, 1.0E-5)
  beta[2] ~ dnorm(0, 1.0E-5)

  # Residual variance
  sigma.e ~ dunif(0, 100)

  # Define prior for the variance-covariance matrix of the random effects
  Omega.u ~ dwish(R, 2)
}"
```

Let's parse this line by line. We'll look at it from the bottom to the top (I don't know why, but everyone on the internet writes their prior distributions at the bottom. It's not intuitive - you want to know the prior before you look at the posterior, right?).

Recall that in a linear mixed effects model (I will refer to this section as the Latex model):

![](https://amolmane1.github.io/images/2017-01-22-Mixed-Models-In-R-latex_model.png)

`Omega.u` is a Wishart distribution. It's a prior on the variance-covariance matrix of the random effects. We need it because in our mixed model, the intercepts of the different teams are normally distributed around their mean (and the same for the slopes of the teams), and this matrix (whose inverse is the precision matrix of the random effects), will be used to calculate just how far the *u*<sub>0*f*</sub> and *u*<sub>1*f*</sub> will be from *β*<sub>0</sub> and *β*<sub>1</sub> respectively. (Refer to lines 10-12 of the code chunk for a better understanding of how `Omega.u` is used to calculate the random effects). `Omega.u` is doing the job of the ∑ in line 2 of the latex model. The values of R are set to be uninformative (the diagonals are .1 because you can't make them 0, because then you won't be able to invert them).

`sigma.e` is an uninformative prior for the residuals. It is used in line 6 of the code chunk to specify the precision of the distribution of `y`. `sigma.e` is doing the job of *σ* in line 3 of the latex model. We do `1/(sigma.e^2)` in line 6 because JAGS syntax takes precision, not standard deviation.

`beta[0]` and `beta[1]` are the priors on the distribution of the two beta values. Again, their values are set to be uninformative - centered at 0 and with a very low precision (so high standard deviation).

Now we've arrived at the data generating model. Lines 10-12 in the code chunk calculate the intercept and `Hits` random effects for all teams (hence the for loop).

Lines 4-7 describe the distribution of y. This is where line 1 of the latex model is being executed.

And that's the JAGS model. Below is how we bring everything together:

``` r
# fit JAGS model with data and model specifications
model <- jags.model(textConnection(model_string),
                    data = model_data,
                    inits = model_inits,
                    n.chains = 10)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 64
    ##    Unobserved stochastic nodes: 8
    ##    Total graph size: 613
    ##
    ## Initializing model

And now we can draw MCMC samples from our model.

``` r
# draw MCMC samples from the model
model_samples = coda.samples(model,
                     variable.names=c("beta", "u"),
                     n.iter= 100000,
                     thin=10)
```

We can take a look at the fixed and random effects calculated by the model below. They look similar to those of the `lmer` model we made.

``` r
# view fixed and random effects
summary(model_samples)[[1]]
```

    ##                Mean        SD   Naive SE Time-series SE
    ## beta[1] 780.0520313 17.769515 0.05619214     0.11161022
    ## beta[2]  75.9007936  9.488034 0.03000380     0.05868819
    ## u[1,1]   -2.1523457 18.892049 0.05974190     0.11123373
    ## u[2,1]  -29.7417014 19.916152 0.06298040     0.14646491
    ## u[3,1]    8.5279237 18.583801 0.05876714     0.11548128
    ## u[4,1]   33.6144092 20.690387 0.06542875     0.17241359
    ## u[1,2]   -0.1613374  8.484282 0.02682966     0.05119793
    ## u[2,2]  -11.9615925 11.300703 0.03573596     0.29894058
    ## u[3,2]    3.6876097  8.571142 0.02710433     0.09016613
    ## u[4,2]   12.5879814 11.289688 0.03570113     0.27702440

To verify that our model is appropriate, and that our sample values have converged, we do some diagnostics below:

``` r
plot(model_samples)
```

Looks like all the beta and random effects values are stable, and the density plots of all the values are approximately normal.

![](https://amolmane1.github.io/images/2017-01-22-Mixed-Models-In-R-diagnostics_1.png)![](https://amolmane1.github.io/images/2017-01-22-Mixed-Models-In-R-diagnostics_2.png)![](https://amolmane1.github.io/images/2017-01-22-Mixed-Models-In-R-diagnostics_3.png)![](https://amolmane1.github.io/images/2017-01-22-Mixed-Models-In-R-diagnostics_4.png)

You could plot the individual fit lines by team for the `lmer` and `JAGS` models to again verify that the results are similar and that the fit lines reasonably approximate the data; however, that's out of the scope of this blog post. My main aim was to help you understand how to code a linear mixed model in JAGS, and I hope I've done that.

Note: code for this post was adapted from <http://bendixcarstensen.com/Bayes/Cph-2012/pracs.pdf>
