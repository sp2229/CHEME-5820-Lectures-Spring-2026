#= 

Linear Models for Classification with Logistic zregression 

Use gradient descent to minimize log-likelihood (cross-entropy function)

Equality, inequality constraints can be used with gradient descent (this can be simplified, in the case of this class)

Update direction is the gradient of the loss fxn w respect to the parameters of the current iteration 
    this is the direction of steepest decrease 
    can be multiplied by learning rate alpha 
    next guess for parameters

Convex/bowl type function 



WHAT ARE THE ISSUES WITH GRADIENT DESCENT: 
    - IF U START WITH A GOOD INITIAL PARAMETER (CLOSE TO ACTUAL ANSWER, GOOD CONVERGENCE)
    - NON-CONVEX system with multiple minimum and maximum, it will converge to one but cannot find the global max/min 
        - fix this by running by multiple areas 


## Summary
This lecture developed logistic regression as a probabilistic model for binary classification, contrasting it with the Perceptron’s hard decision rule.

We derived the logistic function from a Boltzmann distribution with a linear energy model, showing how the inverse temperature beta controls prediction confidence.

Parameter estimation was framed as maximum likelihood, leading to the convex cross-entropy loss optimized by gradient descent and related convergence intuition.

Finally, we introduced regularization (L2 and L1) to curb overfitting, interpret model complexity, and improve generalization.


REGULARIZATION makes the convergence faster by making the bowl steeper 


How to use other ways to optimize: 
    - Diminishing learning rates decreases learning rate over time 
    - momentum methods build velocity by accumulating gradients from previous iterations 


   
   
   
    ## Cross-Validation for Lambda Selection

    Use k-fold cross-validation to find optimal lambda (regularization parameter):
    - Split data into k folds
    - For each lambda value, train on k-1 folds and validate on the held-out fold
    - Calculate average validation error across all folds
    - Select lambda with lowest cross-validation error
    - This prevents overfitting to a single train/test split and ensures robust regularization parameter choice


    Regularization 



    future methods include doing additional capturing of
    much more sophisticarted 

    introduce one vs. rest strategy 