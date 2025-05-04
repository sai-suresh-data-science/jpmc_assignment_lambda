#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 22:46:13 2025

@author: saisuresh
"""

import json
import boto3
import numpy as np
import logging

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Setup SageMaker client
sagemaker_client = boto3.client("sagemaker-runtime", region_name="us-east-1")
ENDPOINT_NAME = "nu-lgbm-infer-endpoint"

def lambda_handler(event, context):
    try:
        logger.info("Received event: %s", event)

        # Parse input
        body = json.loads(event["body"])
        seed = body.get("seed")
        if seed is None:
            logger.warning("Missing seed in request")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'seed' in request body"})
            }

        # Generate 200 vars
        np.random.seed(seed)
        features = np.random.rand(200)
        payload = {f"var_{i}": float(val) for i, val in enumerate(features)}
        logger.info("Payload generated for seed %s", seed)

        # Invoke SageMaker
        response = sagemaker_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read().decode())
        logger.info("Prediction received: %s", result)

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": result})
        }

    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
