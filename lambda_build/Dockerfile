FROM public.ecr.aws/lambda/python:3.12

WORKDIR /lambda_build

# Install packages to /lambda_build
RUN pip install numpy boto3 -t .
