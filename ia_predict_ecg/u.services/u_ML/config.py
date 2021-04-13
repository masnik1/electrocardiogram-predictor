import os


CORS_ORIGIN = os.getenv('CORS_ORIGIN', '*')

PORT = os.getenv('PORT', '8004')

UNIX_OS = os.getenv('UNIX_OS', '0')