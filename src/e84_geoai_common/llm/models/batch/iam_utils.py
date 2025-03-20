import json
from typing import Any

import boto3


def create_role(input_bucket: str, output_bucket: str) -> str:
    """Create the IAM role with S3 access and return its ARN."""
    iam_client: Any = boto3.client("iam")  # type: ignore[reportUnknownMemberType]
    role_name = f"BatchLLMRole-{input_bucket}"
    try:
        # Check if the role already exists.
        role_info = iam_client.get_role(RoleName=role_name)
        return role_info["Role"]["Arn"]
    except iam_client.exceptions.NoSuchEntityException:
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        create_response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for Batch LLM job to access S3 input and output buckets.",
        )
        role_arn = create_response["Role"]["Arn"]
        s3_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{input_bucket}",
                        f"arn:aws:s3:::{output_bucket}",
                    ],
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject"],
                    "Resource": [
                        f"arn:aws:s3:::{input_bucket}/*",
                        f"arn:aws:s3:::{output_bucket}/*",
                    ],
                },
            ],
        }
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="BatchLLM_S3AccessPolicy",
            PolicyDocument=json.dumps(s3_policy),
        )
        return role_arn
