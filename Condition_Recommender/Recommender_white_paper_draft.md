
# Milliman PRM Analytics
## Recommending uncoded conditions

Recent trends in healthcare legislation have lead to a rise in risk-bearing healthcare provider organizations, such as Accountable Care Organizations. Entrusted with the care of thousands of patients, these organizations must leverage data-driven approaches to population health management in order to improve quality of care and reduce costs.

One potential area for gaps in care involves the accuracy of a patient's clinical documentation. Efforts to improve accuracy in a population's clinical records are often referred to as clinical documentation improvement or coding improvement. From a clinical standpoint, the benefit from coding improvement is obvious. A patient record that contains the entirety of the patient's illnesses will result in a more appropriate treatment plan.

However, there are also financial incentives in coding improvement. Alternative payment models often account for the health status of a patient population, through the use of risk scores, when reimbursing a healthcare provider for services. A more accurate clinical record ensures that a risk-bearing healthcare provider is appropriately compensated when they care for a more sick population.

While coding improvement initiatives can focus on conditions that have previously appeared in a medical record, a less obvious approach is to try and identify conditions that have never appeared on a patient's medical record. Traditional approaches for identifying uncoded conditions have used logistic regressions to identify likely comorbidities. At Milliman PRM Analytics, we have taken a unique approach to identifying uncoded conditions through the use of a recommender system. Our recommender system seeks to identify common clinical patterns among patients in a population, and predict conditions that may have previously gone unnoticed.

## What is a recommender system?

If you have ever viewed a product on Amazon or watched a show on Netflix, then you have been a part of a recommender system. Recommender systems are commonly used to help users identify potentially interesting products among a large list of options, through the use of historical viewing or rating information. For example, Netflix will recommend certain shows to you based on your previous viewings. These recommendations are built using viewing or rating data from other users who have viewed the same shows as you.

A common model training process for recommender systems is collaborative filtering, which uses historical rating data to find similarities between users or items. Collaborative filtering often take three forms: user-based, item-based, or matrix factorization. User-based collaborative filtering seeks to find users that have rated items similarly, and predict other items that similar users liked. Item-based collaborative filtering seeks to find similarities between items themselves, and then recommend items that are similar to those that a user rated highly. Matrix factorization collaborative filtering finds similarities between users and items through latent factors, which are then used to composite predicted ratings for each item.

[Collaborative filtering example]

Additionally, the preference inputs in recommender systems may take two forms: explicit ratings or implicit ratings. Explicit ratings are generated when the users themselves identify their preference, such as giving a rating to a movie or a product. While explicit ratings carry a higher level of confidence for a user's preference, they are often not available. More commonly, implicit ratings are inferred from a user's actions, such as viewing a movie or a product.

Our algorithm utilizes an implicit rating, collaborative filtering matrix-factorization model to predict uncoded conditions. Each patient is a "user", with conditions being recommended as the items. Implicit condition confidence values, or ratings, are inferred from the medical history of each patient in a population. These user, condition, and confidence inputs are applied to generate latent factors for each patient and condition. These latent factors, an abstract representation of similarities between users and products, can be combined to generate a predicted rating for each patient-condition pairing. This model has been implemented in Apache Spark, a clustered computing framework.

A matrix factorization recommender system, in many ways, seems like a natural fit for the process of recommending conditions, and offers some advantages over traditional models. This algorithm is fast and simple to train, and thus can realistically be tuned to find unique relationships for each patient population. A recommender system is more patient-focused, and seeks to find top recommendations that are tailored to a patient's unique history. Additionally, a matrix factorization model is able to handle the sparse nature of patient condition information well. Finally, the comorbid nature of many conditions can be expressed well through the use of latent factors in a matrix factorization model.

## Feature Engineering

There are two important considerations for generating useful input data: which features will be used, and how will confidence values for these for these features be determined. Our features are a combination of historical condition information and demographic information. These features and their confidence values are generated from a patient population's clinical history.

For condition features, diagnoses in a patient's clinical history are grouped into clinically meaningful categories, or conditions, using the Agency for Healthcare Research and Quality's (AHRQ's) Clinical Classifications Software (CCS). Patients who are seen for the same condition multiple times are given a higher confidence value. More confidence is given for conditions that have been coded more recently. Additionally, more confidence is given for conditions that were coded in an inpatient setting, rather than an outpatient setting.

The two main demographic features are age and gender. Unlike condition features, demographic features are given the same confidence level across all patients. The confidence value is determined such that demographic importance does not overpower condition information. However, these confidence values must also be large enough that gender-specific and age-specific conditions are modeled appropriately.

## Fitting the Model

The two most important parameters for model selection are lambda, the regularization parameter, and rank, the number of latent factors. Lambda should be tuned to avoid overfitting in the training data, while also still allowing for meaningful variance in predictions. Rank must be selected to allow for meaningful groups of latent factors, while avoiding the computational burden of higher rank models.

Finding an optimal selection of parameters requires a model tuning framework. We would like to determine a model fit which best accomplishes our objective: predicting uncoded conditions. For this purpose, we create a tuning dataset which excludes the most recent months of data. The held out data is analyzed to find conditions that were coded for the first time in a member's medical history. For each model fit, we find each member's top ten recommendations of uncoded conditions. Parameters are chosen from the model fit which recommends the highest number of new conditions in the hold-out set within the top ten predictions.

The hypothetical example below illustrates the process of using latent factors to determine predicted ratings. For simplicity, we will assume a model rank of four, with only four conditions being considered.

[Latent factor dot product example]

[Latent factor dot product explanation]

## Model Performance

There are two characteristics of our model performance that we are interested in: how does this model perform relative to other methods, and how does the model perform when the number of predictions increases. We will examine three models, our predictive model, a popularity-based model for the overall population, and a popularity-based model adjusted for demographics.

The illustration below demonstrates prediction accuracy for our different models as the number of predictions increases. Here we have focused on chronic conditions, as these conditions are more likely to go uncoded if they are not the primary reason that a patient seeks care.

[Illustration placeholder]

The illustration below demonstrates prediction accuracy, now focusing on acute conditions.

[Illustration placeholder]

## Case Study

### Member #1 - Diabetes?

[Confidence for input conditions + ratings of output conditions]

[Explanations for top/interesting prediction]

### Member #2 - Heart Stuff?

[Confidence for input conditions + ratings of output conditions]

[Explanations for top/interesting prediction]

### Member #3 - Female Stuff?

[Confidence for input conditions + ratings of output conditions]

[Explanations for top/interesting prediction]


## Conclusion

Accurately documenting a patient's clinical status will be increasingly important as more healthcare providers enter into alternative payment arrangements. Coding improvement can lead to better clinical outcomes as well as increased revenue in the case of risk-based arrangements. Our recommender system provides a unique perspective towards coding improvement that produces useful recommendations of uncoded conditions.



