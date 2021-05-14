"""
Script that triggers the readthedocs build using a webhook
"""

import os
import subprocess

import requests

webhook_token = os.environ["RTD_WEBHOOK_TOKEN"]
webhook_url = os.environ["RTD_WEBHOOK_URL"]

try:
    git_branch = (subprocess.check_output(["git", "branch", "--show-current"
                                           ]).strip().decode("ascii"))
except subprocess.CalledProcessError:
    raise RuntimeError("can't get git branch")

print(f"Building branch {git_branch}")

# trigger the build
r = requests.post(webhook_url, {
    'branches': git_branch,
    'token': webhook_token
})

if r.status_code != 200:
    raise RuntimeError(f"Couldn't start run")
