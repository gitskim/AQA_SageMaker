import json
import logging
import os
import random
import urllib

import boto3
import numpy as np
import torch
from torchvision import transforms

import cv2 as cv
from models.C3D_altered import C3D_altered
from models.C3D_model import C3D
from models.my_fc6 import my_fc6
from models.score_regressor import score_regressor
from opts import *
import flask

app = flask.Flask(__name__)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info("Loading ProcessPredict function...")

torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True

current_path = os.path.abspath(os.getcwd())

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

s3 = boto3.client("s3")


# PING check used by creation of sagemaker endpoint
@app.route('/ping', methods=['GET'])
def ping_check():
    logger.info("PING!")
    return flask.Response(response=json.dumps({"ping_status": "ok"}), status=200)


'''
curl --data-binary '{"aqa_data": {"bucket_name": "aqauploadprocess-s3uploadbucket-lm9bpkntrclr", "object_key": "20210809_140917_944266.mov"}}' -H "Content-Type: application/json" -v http://localhost:8080/invocations
'''


# Lambda handler executed by lambda function
@app.route('/invocations', methods=['POST', 'PUT'])
def handler():
    logger.info("Received event.")

    data = json.loads(flask.request.data.decode('utf-8'))
    # Get the object from the event and show its content type
    aqa_data = data["aqa_data"]
    bucket = aqa_data["bucket_name"]
    key = aqa_data["object_key"]

    try:
        # Clean up old videos if there is any
        tmp_dir = "/tmp"
        tmp_items = os.listdir(tmp_dir)
        for item in tmp_items:
            if item.endswith(".mov") or item.endswith(".avi") or item.endswith(".mp4"):
                os.remove(os.path.join(tmp_dir, item))

        temp_video_path = f"/tmp/{key}"
        s3.download_file(bucket, key, temp_video_path)
        pred_score = make_prediction(temp_video_path)
        logger.info(f"=== Prediction score: {pred_score} ===")
        response = {"prediction": pred_score}
        response = json.dumps(response)
        return flask.Response(response=response, status=200, mimetype='application/json')
    except Exception as e:
        print(e)
        raise e


# Make prediction core function
def make_prediction(video_file_path):
    val = -1
    if video_file_path:
        frames = preprocess_one_video(video_file_path)
        if frames.shape[2] > 400:
            raise RuntimeError("The uploaded video is too long.")
        preds = inference_with_one_video_frames(frames)
        if preds is None:
            raise RuntimeError("The uploaded video does not seem to be a diving video.")
        val = int(preds[0] * 17)
        logger.info("Predicted score: {}".format(val))
    return val


## Helper functions for processing a video and making prediction
def center_crop(img, dim):
    """Returns center cropped image

    Args:Image Scaling
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2: mid_y + ch2, mid_x - cw2: mid_x + cw2]
    return crop_img


def action_classifier(frames):
    # C3D raw
    model_C3D = C3D()
    model_C3D.load_state_dict(torch.load(c3d_path, map_location={"cuda:0": "cpu"}))
    model_C3D.eval()

    with torch.no_grad():
        X = torch.zeros((1, 3, 16, 112, 112))
        frames2keep = np.linspace(0, frames.shape[2] - 1, 16, dtype=int)
        ctr = 0
        for i in frames2keep:
            X[:, :, ctr, :, :] = frames[:, :, i, :, :]
            ctr += 1
        logger.info(f"X shape: {X.shape}")

        # modifying
        model_C3D.eval()

        # perform prediction
        X = X * 255
        X = torch.flip(X, [1])
        prediction = model_C3D(X)
        prediction = prediction.data.cpu().numpy()

        # print top predictions
        top_inds = prediction[0].argsort()[::-1][
                   :5
                   ]  # reverse sort and take five largest items
        logger.info("\nTop 5:")
        logger.info(f"Top inds: {top_inds}")
    return top_inds[0]


def preprocess_one_video(video_file_path):
    vf = cv.VideoCapture(video_file_path)
    frames = None
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(
            frame, input_resize, interpolation=cv.INTER_LINEAR
        )  # frame resized: (128, 171, 3)
        frame = center_crop(frame, (H, H))
        frame = transform(frame).unsqueeze(0)
        if frames is not None:
            frames = np.vstack((frames, frame))
        else:
            frames = frame

    logger.info(f"frames shape: {frames.shape}")

    vf.release()
    cv.destroyAllWindows()
    rem = len(frames) % 16
    rem = 16 - rem

    if rem != 0:
        padding = np.zeros((rem, C, H, H))
        frames = np.vstack((frames, padding))

    # frames shape: (137, 3, 112, 112)
    frames = torch.from_numpy(frames).unsqueeze(0)

    logger.info(
        f"video shape: {frames.shape}"
    )  # video shape: torch.Size([1, 144, 3, 112, 112])
    frames = frames.transpose_(1, 2)
    frames = frames.double()
    return frames


def inference_with_one_video_frames(frames):
    # load_weights()
    action_class = action_classifier(frames)
    if action_class != 463:
        return None

    model_CNN = C3D_altered()
    model_CNN.load_state_dict(torch.load(m1_path, map_location={"cuda:0": "cpu"}))

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.load_state_dict(torch.load(m2_path, map_location={"cuda:0": "cpu"}))

    # loading our score regressor
    model_score_regressor = score_regressor()
    model_score_regressor.load_state_dict(
        torch.load(m3_path, map_location={"cuda:0": "cpu"})
    )
    with torch.no_grad():
        pred_scores = []

        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()

        clip_feats = torch.Tensor([])
        logger.info(f"frames shape: {frames.shape}")
        for i in np.arange(0, frames.shape[2], 16):
            clip = frames[:, :, i: i + 16, :, :]
            model_CNN = model_CNN.double()
            clip_feats_temp = model_CNN(clip)

            # clip_feats_temp shape: torch.Size([1, 8192])

            clip_feats_temp.unsqueeze_(0)

            # clip_feats_temp unsqueeze shape: torch.Size([1, 1, 8192])

            clip_feats_temp.transpose_(0, 1)

            # clip_feats_temp transposes shape: torch.Size([1, 1, 8192])

            clip_feats = torch.cat((clip_feats.double(), clip_feats_temp), 1)

            # clip_feats shape: torch.Size([1, 1, 8192])

        clip_feats_avg = clip_feats.mean(1)

        model_my_fc6 = model_my_fc6.double()
        sample_feats_fc6 = model_my_fc6(clip_feats_avg)
        model_score_regressor = model_score_regressor.double()
        temp_final_score = model_score_regressor(sample_feats_fc6)
        pred_scores.extend(
            [element[0] for element in temp_final_score.data.cpu().numpy()]
        )

        return pred_scores


def load_weights():
    cnn_loaded = os.path.isfile(m1_path)
    fc6_loaded = os.path.isfile(m2_path)
    score_reg_loaded = os.path.isfile(m3_path)
    div_class_loaded = os.path.isfile(m4_path)
    c3d_loaded = os.path.isfile(c3d_path)
    if cnn_loaded and fc6_loaded and score_reg_loaded and div_class_loaded and c3d_loaded:
        return

    if not cnn_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_CNN, m1_path)
    if not fc6_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_FC6, m2_path)
    if not score_reg_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_SCORE_REG, m3_path)
    if not div_class_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_DIV_CLASS, m4_path)
    if not c3d_loaded:
        urllib.request.urlretrieve(
            "http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle", c3d_path
        )
