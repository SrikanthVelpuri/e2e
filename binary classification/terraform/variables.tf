variable "aws_region" {
  description = "The AWS region to deploy resources in"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "The type of instance to use for the EC2 instance"
  type        = string
  default     = "t2.micro"
}
