# Introduction
  - Changing landscape of healthcare payment
    - Alternative payment models
    - Decline of fee-for-service
    - Need for population health management
  - Importance of coding improvement
    - Financial incentive - increase risk score for alternative payment models
    - Clinical incentive - more appropriate clinical pathways
  - Traditional approach to coding improvement
    - Identify conditions that have been previously coded, but not recently
    - Make sure that conditions have been coded recently if a patient is taking a related drug
  - Brief overview of recommender approach

# What is a recommender algorithm?
  - Traditional uses of recommender algorithm
    - Helping users identify interesting products among many options
    - Ability to scale among a large number of users and/or products
  - Why recommender vs logistic regression?
    - Multi-response nature of recommender (vs 100s of model in logistic)
    - Conditions are often co-morbid and can be expressed well through latent factors
    - Faster and simpler to train than logistic models, so can be tuned on the fly
    - Unique relationships for each population
    - Model perspective is patient-focused, i.e. top N recommendations per patient
    - Sparse nature of patient condition data
  - Types of recommenders
    - User-based collaborative filtering vs Item-based collaborative filtering
    - Explicit ratings vs Implicit ratings
  - Latent factor matrix factorization
    - Simple intuitive example
    - Latent factor dot-product example
  - Type of recommender being implemented here
    - Implicit recommendation
    - Spark/blocked approach?

# Data Collection/Feature Engineering
  - What is a "condition"
    - ~~Determined through claims data/Dx codes~~
    - Mapped into similar conditions using AHRQ CCS
    - ~~Stratified into chronic/non-chronic using AHRQ CCI~~
  - Determining confidence values
    - ~~Exponential decay (acute vs chronic)~~
    - ~~Sum of number of visits~~
    - ~~IP vs OP confidence~~
    - More confidence for recently coded conditions
    - More confidence for IP vs OP
  - Implementing demographic features
    - Age~~, bucketed~~
    - Gender
    - ~~Eligibility/policy group~~
    - Used to better influence latent factors

# Fitting a model
  - Parameters
    - Alpha
    - Lambda
    - Rank
    - (Block size/iterations?)
  - Actual prediction rating calculation
  - Setting up a tuning framework
    - Potential issues with more traditional "testing" framework
      - Holding out users/conditions interferes with cohesive latent factors
    - Holding out a few months of data
    - Evaluating hold-out data by top N prediction methodology

# Model Performance
  - Ignore previously coded conditions
  - Predictive power of 
    - popularity-based model
    - Popularity by age-gender-elig_status?
    - Our algorithm for acute vs chronic conditions
  - Evaluation curve for top N predictions of above models. N->{1:15}?

# Case Study
  - Select some interesting members
  - Show features in + predictions out
  - Demonstrate the "whys" of predictions

# Conclusion
