#!/usr/bin/env bash

git tag --sort=-creatordate --format='%(creatordate:iso-local)%09%(refname:strip=2)' | head -${1:-10}

