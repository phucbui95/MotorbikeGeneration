import boto3
import botocore
import os
import threading
import sys
from boto3.s3.transfer import S3Transfer

class ProgressPercentage(object):
    def __init__(self, client, bucket, object_name, filename):
        self._filename = filename
        self._size = client.head_object(Bucket=bucket,
                                        Key=object_name)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


class S3Client:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def download_file(self, bucket, localFilename, objectKey,
                      display_progress=True):
        """
        Download an object from S3 and save to a local file
        :param bucket: Name of an S3 bucket
        :param localFilename: Local file
        :param objectKey: Object Key
        :return:
        """
        s3 = boto3.resource('s3')
        try:
            if display_progress:
                cb = ProgressPercentage(self.s3, bucket, objectKey, localFilename)
            else:
                cb = None

            transfer = S3Transfer(self.s3)
            transfer.download_file(bucket, objectKey, localFilename,
                                   callback=cb)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

    def upload_file(self, bucket, localFileName, prefix=''):
        """
        Upload a object to an S3 bucket
        :param bucket: Name of the S3 bucket
        :param prefix: Added prefix for object key
        """
        prefix = prefix + '/' if not prefix.endswith('/') else prefix
        Key = prefix + os.path.basename(localFileName)
        self.s3.upload_file(localFileName, bucket, Key)

    def get_matching_s3_objects(self, bucket, prefix='', suffix=''):
        """
        Generate objects in an S3 bucket.

        :param bucket: Name of the S3 bucket.
        :param prefix: Only fetch objects whose key starts with
            this prefix (optional).
        :param suffix: Only fetch objects whose keys end with
            this suffix (optional).
        """
        s3 = self.s3
        kwargs = {'Bucket': bucket}

        # If the prefix is a single string (not a tuple of strings), we can
        # do the filtering directly in the S3 API.
        if isinstance(prefix, str):
            kwargs['Prefix'] = prefix

        while True:

            # The S3 API response is a large blob of metadata.
            # 'Contents' contains information about the listed objects.
            resp = s3.list_objects_v2(**kwargs)

            try:
                contents = resp['Contents']
            except KeyError:
                return

            for obj in contents:
                key = obj['Key']
                if key.startswith(prefix) and key.endswith(suffix):
                    yield obj

            # The S3 API is paginated, returning up to 1000 keys at a time.
            # Pass the continuation token into the next response, until we
            # reach the final page (when this field is missing).
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

    def get_matching_s3_keys(self, bucket, prefix='', suffix=''):
        """
        Generate the keys in an S3 bucket.

        :param bucket: Name of the S3 bucket.
        :param prefix: Only fetch keys that start with this prefix (optional).
        :param suffix: Only fetch keys that end with this suffix (optional).
        """
        for obj in self.get_matching_s3_objects(bucket, prefix, suffix):
            yield obj['Key']

from multiprocessing import Pool, TimeoutError, Process
import time

def upload_fnc(filename, prefix=None):
    if prefix is None:
        prefix = ''
    s3 = S3Client()
    bucket_name = 'phucbb'
    s3.upload_file(bucket_name, filename, prefix=prefix)
    print("DOne")

class S3Storage:
    def __init__(self, name):
        self._processors = []
        self.name = name

    def send_async(self, filename, prefix):
        if prefix is None:
            prefix=''

        p = Process(target=upload_fnc, args=(filename,
                                             self.name + '/' + prefix,))
        p.start()
        self._processors.append(p)

    def __del__(self):
        for p in self._processors:
            try:
                p.join(timeout=20)
            except TimeoutError:
                print("Timeout")

if __name__ == '__main__':

    storage = S3Storage('sample')
    storage.send_async('../download.sh', None)
    time.sleep(10)
    storage.send_async('../Dockerfile', None)


    del storage