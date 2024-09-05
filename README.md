

## GPT-2 Eval

We evaluated the GPT-2 Model on "country" and "contintent" prompts from RAVEL:

### Top-1 Accuracy 

* The accuracy of the model on  evaluation for country is 4.110360360360361%
* The accuracy of the model on  evaluation for continent is 5.546171171171172%

### Top-5 Accuracy

* The accuracy of the model on  evaluation for country is 19.53828828828829%
* The accuracy of the model on  evaluation for continent is 24.46509009009009%

### Top-10 Accuracy

* The accuracy of the model on  evaluation for country is 27.195945945945947%
* The accuracy of the model on  evaluation for continent is 38.65427927927928%

## Mistral Eval

### Top-1 Accuracy

* The accuracy of the model on the evaluation for country is 44%
* The accuracy of the model on  evaluation for continent is 33.671171171171174%

NOTE: One of the bad things about ravel dataset is sometimes, the label is North America, for that even if the prediciton is "north" of the model it is considered as wrong. 


### Accuracy Table

| Evaluation Type | Metric   | Country Accuracy (%)       | Continent Accuracy (%)     |
|-----------------|----------|----------------------------|----------------------------|
| Top-1 Accuracy  | GPT-2 Model   | 4.11                        | 5.55                        |
| Top-5 Accuracy  | GPT-2 Model   | 19.54                       | 24.47                       |
| Top-10 Accuracy | GPT-2 Model   | 27.20                       | 38.65                       |
| Top-1 Accuracy  | Mistral Model | 44.00                       | 33.67                       |



![alt text](image.png)