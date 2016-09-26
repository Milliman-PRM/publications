#Milliman PRM Analytics
##How to measure value in patient care coordination

At-risk organizations such as Accountable Care Organizations have financial stakes in providing high quality care at a reasonable cost. These organizations will need to develop new methods to increase quality while reducing costs. Disease management programs may help improve quality and control costs of individuals with select high-value conditions, but what about everyone else? Milliman PRM Analytics helps care coordination programs properly target individuals likely to incur potentially avoidable medical expenses.

PRM Analytics uses an individual's medical history to predict cost and utilization over the next six months. These predictions can then be used to:

 * identify risks that can be mitigated through care management
 * develop advanced metrics for measuring performance and quality
 * optimize and quantify potential cost saving from care management programs

###Average Joe or Outlier Jane

Prospective risk adjusters are in the business of predicting the cost of an average Joe. The score is commonly determined by applying a matrix of coefficients to medical attributes. This provides the user with clear understanding of the value of medical conditions and how the individual compares to the average Joe in their population. These scores are better at providing insight into a population's health and expenses than the individuals'. In contrast PRM Analytics is built to identify the outlier Janes. The models predict the individual's likelihood of having outlier expenses over the next six months. This focus gives the care coordinators valuable information in both selecting individuals to manage and developing a customized care plan for each individual.

###Let's not call it a black box, maybe gray

PRM Analytics is built upon modern machine learning algorithms. The prediction algorithms are variations of Friedman's gradient boosting machine.[^1] A gradient boosting machine is a frame work for creating boosted decision trees.  A single decision tree is a simple, interpretable model. A series of true/false decision points lead to predictions for each terminal leaf (a node without branches) in the tree. Common prediction metrics include the mean and the median of the data points in each terminal leaf. 

A single decision tree is not competitive with other models on prediction accuracy; you may need to blend decision trees to improve the level of accuracy. Boosting is a method of sequentially building small decision trees to slowly build a more robust and accurate answer. While boosting can often result in dramatic improvements in accuracy, it is at the loss of interpretability and an increased risk of over-fitting.[^2] 

A version of the random subspace method was custom written for PRM Analytics to reduce the models' over-fitting tendencies and improve the computational efficiency.[^3] Random subspacing forces the model to learn from a broader amount of the information gathered for each individual, creating trees with more statistical independence. This results in a more robust model and predictions. 

The custom gradient boosting framework allows for different types of predictions (e.g., probabilities, and averages).  We utilize this framework to make the most useful type of prediction for each prospective outcome:

  - *Probability* of an Inpatient Visit
  - *Probability* of an ER Visit
  - *Potential Outlier Size* of Total Costs
  - *Potential Outlier Size* of Potentially Avoidable Costs 

PRM Analytics uses a repeated *k-fold* cross-validation (*k-fold* CV) framework to tune our models to reduce over-fitting and generalize to the next six months of outcomes. *k*-fold CV involves randomly dividing the set of observations into *k* groups, or *folds*, of approximately equal size. The first fold is treated as a validation set, and the model is built on the remaining *k* − 1 folds. The appropriate error metric is then optimized on the validation set. This procedure is repeated *k* times; each time, a different group of observations is treated as a validation set. This process results in *k* estimates of the test error. The *k-fold* CV estimate is computed by averaging these values.[^4] PRM Analytics not only builds *k* models for each cross-validation fold, the entire process is repeated multiple times. The nature of cross-validation allows for all of the independent models to be built simultaneously in a cloud computing cluster.

###Garbage In, Garbage Out

The Garbage In, Garbage Out adage in our case refers to the need for quality variables or features to make quality predictions. It is important to provide the model with useful information to learn from. The process of developing appropriate data for the model is called feature engineering. A balance must also be reached between too much information and too little information. The historical cost and utilization metrics should be summarized into time periods that are not too long and not too short.  Summarization by month or quarter would generally produce better results than summarization by year or by day. Regardless of these limitations, gradient boosting machines are more robust to data issues than other prediction algorithms. A few of the major advantages are:

 * Weak features will have little influence on the predictions
 * Multicollinearity of variables will not hamper generalization
 * Missing values do not need to be dropped or imputed for processing
 * Non-linearities and interactions between features are natively captured  

Each model is trained using the organization's own medical experience; allowing the algorithms to identify patterns in treatment behavior and billing practices specific each organization. PRM Analytics engineers features for at least the following categories:

1. Demographic - Information about an individual includes:

    * Gender
    * Age
    * Eligibility status
2. Medical conditions - CMS's Hierarchical Condition Categories(HCC)

3. Risk Scores - Appropriate risk scores depend on the line of business.

    * Milliman MARA risk adjuster
    * CMS's HCC Risk Adjustment Model
4. Historical cost and utilization - Aggregations of cost and utilization metrics by,

    * Potentially avoidable services
    * In-network verse Out-of-network services
    * Type of service (e.g., IP, ER, and skilled nursing facility)
    * Date of service

###Conclusion

PRM Analytics produces quality reports to support key roles in risk-bearing organizations. Just as predictive algorithms require quality input, key human decision makers also rely on the quality of their input. The quality output of PRM Analytics will:

 * support care coordination team management and resource optimization,
 * provide tools to identify risks that can be mitigated, and
 * produce advanced metrics for measure provider performance and quality.

###About Milliman

Milliman is among the world's largest providers of actuarial and related products and services. The firm has consulting practices in healthcare, property & casualty insurance, life insurance and financial services, and employee benefits. Founded in 1947, Milliman is an independent firm with offices in major cities around the globe. For further information, visit [milliman.com](http://www.milliman.com/).

[^1]: J.H. Friedman (2001). “[Greedy Function Approximation: A Gradient Boosting Machine](http://projecteuclid.org/download/pdf_1/euclid.aos/1013203451),” Annals of Statistics 29(5):1189-1232.

[^2]: G. James et al., An Introduction to Statistical Learning: with Applications in R, Springer Texts in Statistics 103, p. 303 DOI [10.1007/978-1-4614-7138-7](http://dx.doi.org/10.1007%2F978-1-4614-7138-7)

[^3]: Ho, Tin (1998). "The Random Subspace Method for Constructing Decision Forests". IEEE Transactions on Pattern Analysis and Machine Intelligence 20 (8): 832–844. doi:[10.1109/34.709601.](http://dx.doi.org/10.1109%2F34.709601)

[^4]: G. James et al., An Introduction to Statistical Learning: with Applications in R, Springer Texts in Statistics 103, p. 181 DOI [10.1007/978-1-4614-7138-7](http://dx.doi.org/10.1007%2F978-1-4614-7138-7)
