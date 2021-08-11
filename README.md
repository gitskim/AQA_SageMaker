# AQA_SageMaker

## To test with Flask

1. under AQA_SageMaker directory
```c
export FLASK_APP=process_predict.py
flask run
```

2. open another terminal
3. run the following

```
curl --data-binary '{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lm9bpkntrclr", "object_key": "20210809_140917_944266.mov"}}' -H "Content-Type: application/json" -v http://localhost:5000/invocations

```

## To test with docker
1. 
```
docker build . -t aqa
```
2. 
``` 
docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm aqa serve
```
3. Open another terminal
4. Run the following
```
curl --data-binary '{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lm9bpkntrclr", "object_key": "20210809_140917_944266.mov"}}' -H "Content-Type: application/json" -v http://localhost:8080/invocations

```

## Note
Only the port number is different when I opened another terminal to test. 
