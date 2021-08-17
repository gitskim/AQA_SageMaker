from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import json

role = get_execution_role()

aqa_model = Model(
    image_uri="724670356997.dkr.ecr.us-west-2.amazonaws.com/aqa-predict:latest",
    role=role
)

predictor = aqa_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

pred_336 = Predictor("aqa-predict-2021-08-12-11-47-51-336", serializer=JSONSerializer(), deserializer=JSONDeserializer())

pred_160 = Predictor("aqa-predict-2021-08-12-11-53-39-160", serializer=JSONSerializer(), deserializer=JSONDeserializer())

data = """{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lm9bpkntrclr", "object_key": "20210809_140917_944266.mov"}}"""
output = pred_336.predict(json.loads(data))