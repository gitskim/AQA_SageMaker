# AQA_SageMaker

Based on the tutorial [Deploying your ML models to AWS SageMaker
](https://towardsdatascience.com/deploying-your-ml-models-to-aws-sagemaker-6948439f48e1).

The following example starts from here: https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own/container

## To test with docker locally
* Make sure to include the models and their weights in under the models directory.

1. Build the Docker container:
```
docker build . -t aqa
```
2. Run the Docker container
``` 
docker run -e AWS_ACCESS_KEY_ID=<your_aws_access_key_id> -e AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key> -v $(pwd) -p 8080:8080 --rm aqa serve
```
3. Open another terminal
4. Send a request to the server
```
curl --data-binary '{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lntb7jdvnhhg", "object_key": "20211117_102853_978476.mp4"}}' -H "Content-Type: application/json" -v http://localhost:8080/invocations

```
