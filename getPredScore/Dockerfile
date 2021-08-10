FROM public.ecr.aws/lambda/python:3.7

# COPY __init__.py app.py requirements.txt opts.py ./
COPY . /getPredScore
WORKDIR /getPredScore

RUN python3.7 -m pip install -r requirements.txt -t .
ENV PYTHONPATH "${PYTHONPATH}:/getPredScore/"
RUN ls


# Command can be overwritten by providing a different command in the template directly.
#CMD ["process_predict.handler"]
