import boto3
import threading
from s3_client import S3Client

def upload_file(filename):
    s3 = S3Client()
    bucket_name = 'phucbb'
    s3.upload_file(bucket_name, filename, prefix='experiments/')
    print("Done")

if __name__ == '__main__':
    filename = '../generator_epoch0.pth'

    t = threading.Thread(target=upload_file, args=(filename, ))
    t.start()

    print("hello world")




