#!/bin/bash

set -eo pipefail

if [ -z "$DATA_PASSWORD" ]; then
	echo -n "Password: "
	read -s DATA_PASSWORD
	echo
fi

DATA_URL=${DATA_URL:-https://big-tank.app.tu-dortmund.de/pyirf-testdata/}

wget \
	-R "*.html*,*.gif" \
	--no-host-directories --cut-dirs=1 \
	--no-parent \
	--user=pyirf \
	--password="$DATA_PASSWORD" \
	--no-verbose \
	--recursive \
	--directory-prefix=data \
	"$DATA_URL"
