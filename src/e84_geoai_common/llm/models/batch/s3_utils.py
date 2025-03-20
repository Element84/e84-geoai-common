import json
from typing import Any

from botocore.exceptions import ClientError

# ruff: noqa: N815, N803, N806


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return the bucket name and key from an S3 URI."""
    if not uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI; must start with 's3://'.")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def ensure_bucket_exists(s3_client: Any, bucket: str, *, create_buckets_if_missing: bool) -> None:  # noqa: ANN401
    """Ensure that the specified bucket exists and creates it if it doesn't and flag is true."""
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        if create_buckets_if_missing:
            region = s3_client.meta.region_name
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket)
            else:
                s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
        else:
            msg = f"Bucket {bucket} does not exist and create_buckets_if_missing flag is False."
            raise ValueError(msg) from e


def load_file_from_s3(s3_client: Any, bucket: str, key: str) -> str:  # noqa: ANN401
    """Load a file from the specified S3 bucket and key, returning its body as a string."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    return body


def save_role_arn(s3_client: Any, bucket: str, roleArn: str) -> None:  # noqa: ANN401
    """Save the role ARN to the input S3 bucket at the root in role_arn.json."""
    key = "role_arn.json"
    body = json.dumps({"roleArn": roleArn})
    s3_client.put_object(Bucket=bucket, Key=key, Body=body)


def load_role_arn(s3_client: Any, bucket: str) -> str:  # noqa: ANN401
    """Load the role ARN from role_arn.json in the root of the input S3 bucket."""
    key = "role_arn.json"
    body = load_file_from_s3(s3_client, bucket, key)
    data = json.loads(body)
    role_arn = data.get("roleArn")
    if not role_arn:
        raise ValueError("roleArn not found in the file.")
    return role_arn


def save_job_arn(s3_client: Any, bucket: str, jobName: str, jobArn: str) -> None:  # noqa: ANN401
    """Save the job ARN to a JSON file in the input S3 bucket.

    The file is saved under <jobName>/job_arn.json.
    """
    key = f"{jobName}/job_arn.json"
    body = json.dumps({"jobArn": jobArn})
    s3_client.put_object(Bucket=bucket, Key=key, Body=body)


def load_job_arn(s3_client: Any, bucket: str, jobName: str) -> str:  # noqa: ANN401
    """Load the job ARN from the JSON file in the input S3 bucket.

    The file should be under <jobName>/job_arn.json.
    """
    key = f"{jobName}/job_arn.json"
    body = load_file_from_s3(s3_client, bucket, key)
    data = json.loads(body)
    job_arn = data.get("jobArn")
    if not job_arn:
        raise ValueError("Job ARN not found in S3 object.")
    return job_arn
