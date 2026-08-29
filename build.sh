#!/usr/bin/env bash
# Exit on error
set -o errexit

# Nâng cấp pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
