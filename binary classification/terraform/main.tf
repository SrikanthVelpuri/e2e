provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = var.instance_type

  tags = {
    Name = "ExampleInstance"
  }
}

resource "aws_s3_bucket" "example" {
  bucket = "example-bucket-${random_id.bucket_id.hex}"
  acl    = "private"

  tags = {
    Name        = "ExampleBucket"
    Environment = "Dev"
  }
}

resource "random_id" "bucket_id" {
  byte_length = 8
}
