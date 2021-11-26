# AQA_SageMaker

The following example starts from here: https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own/container

## To test with Flask

1. under AQA_SageMaker directory
```c
export FLASK_APP=process_predict.py
flask run
```

2. open another terminal
3. run the following

```
curl --data-binary '{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lntb7jdvnhhg", "object_key": "20211117_102853_978476.mp4"}}' -H "Content-Type: application/json" -v http://localhost:5000/invocations

```

## To test with docker
1. 
```
docker build . -t aqa
```
2. 
``` 
docker run -e AWS_ACCESS_KEY_ID=<your_aws_access_key_id> -e AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key> -v $(pwd) -p 8080:8080 --rm aqa serve
```
3. Open another terminal
4. Run the following
```
curl --data-binary '{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lntb7jdvnhhg", "object_key": "20211117_102853_978476.mp4"}}' -H "Content-Type: application/json" -v http://localhost:8080/invocations

```

## Note
Only the port number is different when I opened another terminal to test. 
