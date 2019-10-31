import boto3

if __name__ == '__main__':
    # Create an S3 client
    s3 = boto3.client('s3')

    filename = '../generator_epoch0.pth'
    bucket_name = 'phucbb'

    # Uploads the given file using a managed uploader, which will split up large
    # files automatically and upload parts in parallel.
    source_filename = filename
    key_name = filename.split('/')[-1]

    s3.upload_file(source_filename, bucket_name, key_name)