# Re-rank Simulator

## Setup
  1. Python 3.6
  2. TensorFlow 1.12

## Environment
  * Env.Documents 
    * Generate N documents feature randomly.
  * Env.UserResponse
    * Build a user simulator.

## DataSet
  * Get train/validation/test set
    ```
    $ python dataset.py
    ```

## Demo
  * Model training 
  ```
  $ python dnn_train.py --algo PointWise --loss ce --timestamp 20201010101010
  ```
  * Model testing
  ```
  $ python dnn_test.py --algo PointWise --loss ce --timestamp 20201010101010
  ``` 

## Metrics
  * GAUC
  * NDCG
  * Evaluator score is the prediction of the evaluator. 
  * True score is the feedback of the user simulator. 